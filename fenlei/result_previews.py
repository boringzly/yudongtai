"""Bounded, task-level QA contact sheets. Never renders every tile or polygon.

All geospatial/image dependencies are lazy: optional QA must not prevent a task
from delivering its vector result. A sheet displays ONE sampled result polygon.
"""

import hashlib
import heapq
import html
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

from classification_schema import CLASS_NAME_BY_CODE

GENERATOR = 'classification_result_previews_v1'
DEFAULT_COUNT = 30
MAX_COUNT = 60
MAX_BYTES = 50 * 1024 * 1024
PANEL_SIZE = 480
COLORS = [(130, 140, 151), (235, 190, 58), (51, 142, 82), (143, 205, 81),
          (57, 153, 239), (228, 96, 78), (186, 156, 124)]
EN_NAMES = ['Unknown', 'Cropland', 'Forest', 'Grassland', 'Water', 'Built-up', 'Unused land']


def preview_count(environ=None):
    """0 disables previews; invalid settings fall back to the conservative default."""
    environ = os.environ if environ is None else environ
    try:
        return max(0, min(MAX_COUNT, int(environ.get('CLASSIFICATION_PREVIEW_COUNT', DEFAULT_COUNT))))
    except (TypeError, ValueError):
        return DEFAULT_COUNT


def _code(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if 0 <= value <= 6 else 0


def _priority(text):
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'big')


def _round_robin(groups, count):
    result = []
    for index in range(count):
        for group in groups:
            if index < len(group):
                result.append(group[index])
                if len(result) == count:
                    return result
    return result


def sample_features(sources, count, warn, progress):
    """Streaming, repeatable stratification by source pair and class transition.

    Only FIDs/properties are retained, not full geometries. At most count source
    groups survive, each with count candidates. No dependence on input feature
    order or an early concentration of one class such as water.
    """
    if count <= 0:
        return []
    import fiona

    pair_groups = []
    for source in sources:
        progress('正在抽样图斑: ' + Path(source['shp']).stem)
        # Basename keeps selections stable when the task output root changes.
        key = Path(source['shp']).name
        buckets = {}
        try:
            with fiona.open(source['shp']) as collection:
                if not collection.crs_wkt and not collection.crs:
                    raise ValueError('SHP 缺少坐标系，无法安全叠加')
                for index, feature in enumerate(collection):
                    if index and index % 10000 == 0:
                        progress('正在抽样图斑: %s，已扫描 %d 个' % (key, index))
                    if feature['geometry'] is None:
                        continue
                    props = feature['properties']
                    transition = (_code(props.get('pre_code')), _code(props.get('curr_code')))
                    fid = str(feature['id'])
                    rank = _priority('42|%s|%s' % (key, fid))
                    bucket = buckets.setdefault(transition, [])
                    item = (-rank, fid)
                    if len(bucket) < count:
                        heapq.heappush(bucket, item)
                    elif item > bucket[0]:
                        heapq.heapreplace(bucket, item)
            groups = []
            for transition in sorted(buckets, key=lambda t: _priority(key + str(t))):
                groups.append([{'source': source, 'fid': fid} for _, fid in sorted(buckets[transition], reverse=True)])
            candidates = _round_robin(groups, count)
            if candidates:
                pair_groups.append((_priority(key), key, candidates))
                pair_groups.sort(key=lambda item: (item[0], item[1]))
                del pair_groups[count:]
        except Exception as exc:
            warn('抽样失败 %s: %s' % (source['shp'], exc))
    return _round_robin([group[2] for group in pair_groups], count)


def _font_path():
    paths = [os.environ.get('CLASSIFICATION_PREVIEW_FONT', ''),
             '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
             '/usr/share/fonts/truetype/arphic/uming.ttc',
             '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc',
             'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']
    return next((path for path in paths if path and Path(path).is_file()), None)


def _read_grid(path, crs, transform, size, is_rgb=False):
    """Warp directly to the small shared preview grid; never read a full TIFF."""
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp, Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError('影像缺少坐标系: ' + str(path))
        if is_rgb:
            interpretation = list(src.colorinterp)
            rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            if all(color in interpretation for color in rgb):
                bands = [interpretation.index(color) + 1 for color in rgb]
            elif src.count >= 3:
                bands = [1, 2, 3]
            else:
                bands = [1, 1, 1]
        else:
            bands = [1]
        with WarpedVRT(src, crs=crs, transform=transform, width=size, height=size,
                       resampling=Resampling.bilinear if is_rgb else Resampling.nearest,
                       add_alpha=ColorInterp.alpha not in src.colorinterp,
                       warp_mem_limit=32) as vrt:
            values = vrt.read(bands, masked=True)
            valid = ~np.any(np.ma.getmaskarray(values), axis=0)
            values = values.filled(0)
            valid &= np.all(np.isfinite(values), axis=0)
    return values, valid


def _rgb_pair(before, after):
    """Keep byte RGB true-colour; jointly stretch higher bit depths, never per-date."""
    import numpy as np

    if before[0].dtype == np.uint8 and after[0].dtype == np.uint8:
        outputs = []
        for array, valid in (before, after):
            image = array.transpose(1, 2, 0).copy()
            image[~valid] = (32, 40, 52)
            outputs.append(image)
        return outputs
    outputs = []
    ranges = []
    for channel in range(3):
        samples = [array[channel][valid] for array, valid in (before, after)]
        samples = [values for values in samples if values.size]
        if samples:
            low, high = np.percentile(np.concatenate(samples), [2, 98])
            if high <= low:
                high = low + 1
        else:
            low, high = 0, 255
        ranges.append((low, high))
    for array, valid in (before, after):
        image = np.zeros((*valid.shape, 3), dtype='uint8')
        for channel, (low, high) in enumerate(ranges):
            safe = np.nan_to_num(array[channel].astype('float32'), nan=low, posinf=high, neginf=low)
            image[:, :, channel] = (np.clip((safe - low) / (high - low), 0, 1) * 255).astype('uint8')
        image[~valid] = (32, 40, 52)
        outputs.append(image)
    return outputs


def _coarse_grid(raw, valid, class_mapping):
    import numpy as np

    result = np.zeros(raw.shape, dtype='uint8')
    for original, coarse in class_mapping.items():
        if coarse:
            result[(raw == original) & valid] = coarse
    return result


def _blend(rgb, mask, color, alpha):
    import numpy as np

    result = rgb.copy()
    result[mask] = ((1 - alpha) * result[mask] + alpha * np.asarray(color)).astype('uint8')
    return result


def _pixel_overlay(rgb, classes, valid):
    import numpy as np

    palette = np.asarray(COLORS, dtype='uint8')[classes]
    result = rgb.copy()
    selected = valid & (classes > 0)
    result[selected] = (result[selected] * .35 + palette[selected] * .65).astype('uint8')
    return result


def _outline(image, geometry, transform, color=(255, 225, 105), width=2):
    """Draw exterior AND hole boundaries, including subpixel diagonal roads."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
    for polygon in polygons:
        for ring in polygon:
            points = [(~transform) * (point[0], point[1]) for point in ring]
            if len(points) >= 2:
                draw.line(points, fill=(18, 24, 32), width=width + 2)
                draw.line(points, fill=color, width=width)
    return image


def render_sheet(candidate, number, class_mapping):
    import fiona
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw, ImageFont
    from rasterio.features import bounds as geometry_bounds, rasterize
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_geom

    source = candidate['source']
    size = PANEL_SIZE
    with fiona.open(source['shp']) as collection:
        feature = collection[int(candidate['fid'])]
        if feature is None:
            raise ValueError('抽样图斑已不存在')
        props = dict(feature['properties'])
        vector_crs = collection.crs_wkt or collection.crs
        with rasterio.open(source['post_image']) as reference:
            crs = reference.crs
            if crs is None:
                raise ValueError('后时相缺少坐标系')
            geometry = transform_geom(vector_crs, crs, feature['geometry'])
            pixel_size = max(reference.res)
    if geometry['type'] not in ('Polygon', 'MultiPolygon'):
        raise ValueError('预览仅支持面图斑')
    left, bottom, right, top = geometry_bounds(geometry)
    if not all(math.isfinite(value) for value in (left, bottom, right, top)):
        raise ValueError('图斑范围无效')
    side = max(right - left, top - bottom, pixel_size * 128) * 1.4
    if side <= 0:
        raise ValueError('图斑范围为空')
    x, y = (left + right) / 2, (bottom + top) / 2
    bbox = (x - side / 2, y - side / 2, x + side / 2, y + side / 2)
    transform = from_bounds(*bbox, size, size)
    mask = rasterize([(geometry, 1)], out_shape=(size, size), transform=transform,
                     all_touched=False, fill=0, dtype='uint8').astype(bool)
    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024):
        before = _read_grid(source['pre_image'], crs, transform, size, True)
        after = _read_grid(source['post_image'], crs, transform, size, True)
        if not np.any(before[1]) or not np.any(after[1]):
            raise ValueError('抽样窗口在某一时相无有效影像像元，无法进行前后对照')
        rgb_pre, rgb_post = _rgb_pair(before, after)
        pixel_images = []
        for key, rgb in [('pre_class', rgb_pre), ('post_class', rgb_post)]:
            raw, valid = _read_grid(source[key], crs, transform, size)
            classes = _coarse_grid(raw[0], valid, class_mapping)
            pixel_images.append(_pixel_overlay(rgb, classes, valid))

    pre_code, post_code = _code(props.get('pre_code')), _code(props.get('curr_code'))
    font_file = _font_path()
    if font_file:
        try:
            ImageFont.truetype(font_file, 19)
        except OSError:
            font_file = None
    chinese = bool(font_file)
    # CJK-less containers remain readable; the HTML/manifest always use Chinese.
    if not font_file:
        font_file = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def font(size):
        try:
            return ImageFont.truetype(font_file, size)
        except OSError:
            return ImageFont.load_default()
    heading, normal, small = font(23), font(19), font(16)
    names = [CLASS_NAME_BY_CODE[i] for i in range(7)] if chinese else EN_NAMES
    titles = (['01  前时相原图', '02  后时相原图', '03  后时相 + 变化边界',
               '04  前时相像元分类（六大类）', '05  后时相像元分类（六大类）', '06  后时相 + 变化区域',
               '07  最终图斑 · 前时相类别', '08  最终图斑 · 后时相类别', '09  抽样信息 / 分类图例'] if chinese else
              ['01  Before / imagery', '02  After / imagery', '03  After / change outline',
               '04  Before / pixel classes', '05  After / pixel classes', '06  After / change area',
               '07  Final polygon / before', '08  Final polygon / after', '09  Sample / legend'])
    images = [Image.fromarray(rgb_pre), Image.fromarray(rgb_post),
              _outline(Image.fromarray(rgb_post), geometry, transform),
              _outline(Image.fromarray(pixel_images[0]), geometry, transform),
              _outline(Image.fromarray(pixel_images[1]), geometry, transform),
              _outline(Image.fromarray(_blend(rgb_post, mask, (255, 214, 73), .35)), geometry, transform),
              _outline(Image.fromarray(_blend(rgb_pre, mask, COLORS[pre_code], .6)), geometry, transform, COLORS[pre_code], 3),
              _outline(Image.fromarray(_blend(rgb_post, mask, COLORS[post_code], .6)), geometry, transform, COLORS[post_code], 3)]
    gutter, header, title_height, footer = 16, 94, 42, 54
    width, height = 3 * size + 4 * gutter, header + 3 * (size + title_height + gutter) + footer
    sheet = Image.new('RGB', (width, height), (16, 24, 37))
    draw = ImageDraw.Draw(sheet)
    def fit(text, max_width, selected_font):
        while text and draw.textbbox((0, 0), text, font=selected_font)[2] > max_width:
            text = text[:-4] + '...' if len(text) > 4 else ''
        return text
    title = ('变化检测与分类 · 抽样质检' if chinese else 'CHANGE + CLASSIFICATION / SAMPLE REVIEW')
    draw.text((gutter, 12), title + '  #%02d' % number, font=heading, fill=(243, 247, 254))
    subtitle = Path(source['pre_image']).name + '  ->  ' + Path(source['post_image']).name
    if not chinese:
        subtitle = subtitle.encode('ascii', 'replace').decode('ascii')
    draw.text((gutter, 49), fit(subtitle, width - 2 * gutter, normal), font=normal, fill=(154, 173, 195))
    for i, title in enumerate(titles):
        px = gutter + (i % 3) * (size + gutter)
        py = header + (i // 3) * (size + title_height + gutter)
        draw.rectangle((px, py, px + size - 1, py + title_height + size - 1), fill=(27, 40, 57))
        draw.text((px + 12, py + 9), title, font=normal, fill=(234, 241, 252))
        if i < 8:
            sheet.paste(images[i], (px, py + title_height))
        else:
            text_x, line_y = px + 18, py + title_height + 14
            uid = str(props.get('uid', candidate['fid']))
            lines = [('图斑 UID: ' if chinese else 'Polygon UID: ') + uid,
                     '%s %s  ->  %s %s' % (pre_code, names[pre_code], post_code, names[post_code]),
                     ('同类变化，建议重点核查' if chinese else 'Same class: inspect visually') if pre_code == post_code else
                     ('类别发生变化' if chinese else 'Class transition'),
                     ('黄色边界仅标出本抽样图斑' if chinese else 'Outline: this sampled polygon only')]
            for line in lines:
                draw.text((text_x, line_y), line, font=normal, fill=(220, 233, 247))
                line_y += 32
            line_y += 12
            for code, name in enumerate(names):
                draw.rectangle((text_x, line_y + 4, text_x + 17, line_y + 21), fill=COLORS[code])
                draw.text((text_x + 30, line_y), '%d  %s' % (code, name), font=normal, fill=(220, 233, 247))
                line_y += 31
            for line in (str(crs), ('中心: ' if chinese else 'Center: ') + '%.3f, %.3f' % (x, y),
                         ('预览已降采样，不用于精确测量' if chinese else 'Downsampled QA, not a measurement')):
                draw.text((text_x, line_y + 8), fit(line, size - 36, small), font=small, fill=(154, 173, 195))
                line_y += 26
    note = ('同范围 / 同比例 · 前后统一显示规则 · 类别颜色见图例 · 仅为抽样，不代表总体精度' if chinese else
            'Matched extent / scale | Matched display rules | Samples, not an accuracy assessment')
    draw.text((gutter, height - 37), note, font=normal, fill=(154, 173, 195))
    output = io.BytesIO()
    sheet.save(output, format='JPEG', quality=85, optimize=True)
    info = {'fid': candidate['fid'], 'uid': props.get('uid'), 'pre_code': pre_code, 'curr_code': post_code,
            'pre_name': CLASS_NAME_BY_CODE[pre_code], 'curr_name': CLASS_NAME_BY_CODE[post_code],
            'source': {key: str(value) for key, value in source.items()}, 'crs': str(crs),
            'bounds': list(bbox), 'chinese_font': chinese,
            'valid_image_fraction': [float(np.mean(before[1])), float(np.mean(after[1]))]}
    return output.getvalue(), info


def _prepare_directory(directory):
    """Only remove files recorded by our previous manifest, never user files."""
    if directory.is_symlink():
        raise ValueError('预览目录不能是符号链接')
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).is_symlink() for name in ('manifest.json', 'index.html', 'manifest.json.tmp')):
        raise ValueError('预览清单或总览页不能是符号链接')
    manifest = directory / 'manifest.json'
    existing = []
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('generator') != GENERATOR:
            raise ValueError('预览目录已有非本程序的 manifest.json，拒绝覆盖')
        existing = [item['file'] for item in previous.get('samples', [])]
    elif any(directory.iterdir()):
        raise ValueError('预览目录非空且无本程序清单，拒绝覆盖')
    for name in existing:
        if re.fullmatch(r'sample_[0-9]{3}\.jpg', name):
            target = directory / name
            if target.resolve().parent == directory.resolve() and not target.is_symlink() and target.is_file():
                target.unlink()


def _write_index(directory, report):
    cards = []
    for item in report['samples']:
        title = '%s：%s → %s（UID %s）' % (item['file'], item['pre_name'], item['curr_name'], item['uid'])
        cards.append('<a href="%s"><img loading="lazy" src="%s"><p>%s</p></a>' %
                     (item['file'], item['file'], html.escape(title)))
    warning = '；'.join(report['warnings'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>变化检测与分类 · 抽样质检</title>
<style>body{background:#101825;color:#eaf1fc;font:16px sans-serif;margin:32px}p{color:#b5c6da}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}
a{color:inherit;text-decoration:none;background:#1b2839;padding:12px;border-radius:8px}img{width:100%%}</style>
<h1>变化检测与分类 · 抽样质检</h1><p>本次 %d 张，最多 %d 张。点击查看完整九宫格。</p>
<p>兼顾不同影像与类别转换；不是面积比例抽样，不能用来估算总体精度或类别频率。黄色边界仅标出所抽图斑。</p>
<p>%s</p><main>%s</main></html>''' % (len(report['samples']), report['requested_count'], html.escape(warning), ''.join(cards))
    (directory / 'index.html').write_text(page, encoding='utf-8')
    temp = directory / 'manifest.json.tmp'
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(directory / 'manifest.json')


def generate_previews(sources, dst_path, class_mapping, logger, progress=None):
    """Best-effort task hook. Returns a diagnostic summary, never fails the SHP."""
    count = preview_count()
    report = {'generator': GENERATOR, 'status': 'disabled' if count == 0 else 'completed',
              'requested_count': count, 'samples': [], 'warnings': [], 'bytes': 0}
    directory = Path(dst_path).resolve() / 'previews'
    if count == 0:
        logger.info('抽样预览已关闭 (CLASSIFICATION_PREVIEW_COUNT=0)')
        # Reusing an output root must not show the previous run's samples as new.
        if (directory / 'manifest.json').exists():
            try:
                _prepare_directory(directory)
                _write_index(directory, report)
            except Exception as exc:
                report['warnings'].append(str(exc))
                logger.warning('关闭预览时清理旧预览失败: %s', exc)
        return report
    def warn(message):
        logger.warning('抽样预览: %s', message)
        if len(report['warnings']) < 20:
            report['warnings'].append(message)
    def notify(message):
        logger.info('抽样预览: %s', message)
        if progress:
            try:
                progress(message)
            except Exception:
                logger.warning('预览进度发送失败', exc_info=True)
    prepared = False
    try:
        _prepare_directory(directory)
        prepared = True
        # Clear the previous run's gallery immediately, even if rendering fails.
        _write_index(directory, report)
        candidates = sample_features(sources, count, warn, notify)
        for index, candidate in enumerate(candidates, 1):
            notify('正在生成抽样九宫格 %d/%d' % (index, len(candidates)))
            try:
                data, info = render_sheet(candidate, len(report['samples']) + 1, class_mapping)
                if report['bytes'] + len(data) > MAX_BYTES:
                    warn('已达到预览 50 MiB 总大小上限，停止生成')
                    break
                info['file'] = 'sample_%03d.jpg' % (len(report['samples']) + 1)
                target = directory / info['file']
                if target.exists():
                    raise ValueError('目标预览文件已存在，拒绝覆盖未登记文件: ' + str(target))
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(prefix='.preview-', suffix='.tmp', dir=directory, delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(data)
                    temporary.replace(target)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
                report['samples'].append(info)
                report['bytes'] += len(data)
                # Persist ownership after each sheet, so retry doesn't keep stale images.
                _write_index(directory, report)
            except Exception as exc:
                warn('生成失败 %s FID=%s: %s' % (candidate['source']['shp'], candidate['fid'], exc))
        if not candidates and not report['warnings']:
            warn('本次无可抽样变化图斑，没有生成图片；请结合分类日志判断是否为有效空结果')
        if report['warnings']:
            report['status'] = 'completed_with_warnings'
        if any(not item['chinese_font'] for item in report['samples']):
            logger.info('服务器未找到中文字体，预览图使用英文标签；总览页与清单仍为中文。可设置 CLASSIFICATION_PREVIEW_FONT')
        _write_index(directory, report)
        report['directory'] = str(directory)
        report['index'] = str(directory / 'index.html')
        logger.info('抽样预览完成: %d 张，%.1f MiB；总览: %s',
                    len(report['samples']), report['bytes'] / 1024**2, report['index'])
    except Exception as exc:
        report['status'] = 'failed'
        warn('预览失败，SHP 结果不受影响: %s' % exc)
        if prepared:
            try:
                _write_index(directory, report)
            except Exception:
                logger.warning('预览清单无法写入', exc_info=True)
    return report
