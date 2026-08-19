import requests
import json

def test_gdb_api():
    """测试GDB API接口"""
    
    # API参数
    params = {
        'ak': 'mf85056077e36b72b8ef9170acd8d95b9e',
        'op': 'select_with_ref',
        'page_count': 200000,
        'page_num': 1,
        'file_type': 'gdb',
        'release_status': 'online',
        'usage_status': 'ready',
        'admin_province_code': 130000,
        'release_year': 2024,
        'type': 'current'
    }
    
    # API地址
    api_url = "http://172.20.46.51:7010/sj_assets/v6/api/ecomn/raster_result"
    
    print("=" * 50)
    print("测试GDB API接口")
    print("=" * 50)
    print(f"API地址: {api_url}")
    print(f"请求参数: {json.dumps(params, indent=2)}")
    print("-" * 50)
    
    try:
        # 发送请求
        print("发送请求...")
        response = requests.get(api_url, params=params, timeout=30)
        
        # 检查HTTP状态码
        print(f"HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            # 尝试解析JSON响应
            try:
                data = response.json()
                print("✓ 成功获取JSON响应")
                
                # 检查返回数据结构
                if 'result' in data:
                    result = data['result']
                    print(f"返回结果字段:")
                    print(f"  - page_num: {result.get('page_num')}")
                    print(f"  - page_count: {result.get('page_count')}")
                    print(f"  - total_count: {result.get('total_count')}")
                    
                    # 检查item_list
                    if 'item_list' in result and result['item_list']:
                        item_list = result['item_list']
                        print(f"  - item_list: 找到 {len(item_list)} 个项目")
                        
                        # 打印每个项目的关键信息
                        for i, item in enumerate(item_list):
                            print(f"\n项目 {i+1}:")
                            print(f"   名称: {item.get('name', 'N/A')}")
                            print(f"   文件路径: {item.get('file_path', 'N/A')}")
                            print(f"   数据ID: {item.get('data_id', 'N/A')}")
                            print(f"   发布年份: {item.get('release_year', 'N/A')}")
                            print(f"   省级编码: {item.get('admin_province_code', 'N/A')}")
                    else:
                        print("  - item_list: 空列表")
                        
                else:
                    print("✗ 响应中缺少 'result' 字段")
                    print(f"完整响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    
            except json.JSONDecodeError as e:
                print(f"✗ JSON解析失败: {e}")
                print(f"原始响应内容: {response.text}")
                
        else:
            print(f"✗ HTTP请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectTimeout:
        print("✗ 连接超时，请检查网络或服务器地址")
    except requests.exceptions.ConnectionError:
        print("✗ 连接错误，请检查服务器是否可访问")
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")
    except Exception as e:
        print(f"✗ 未知错误: {e}")
    
    print("=" * 50)


def test_multiple_provinces():
    """测试多个省份编码"""
    
    test_cases = [
        {'admin_province_code': 440000, 'release_year': 2024, 'description': '广东省2024年'},
        {'admin_province_code': 440000, 'release_year': 2023, 'description': '广东省2023年'},
        {'admin_province_code': 110000, 'release_year': 2024, 'description': '北京市2024年'},
        {'admin_province_code': 310000, 'release_year': 2024, 'description': '上海市2024年'},
    ]
    
    api_url = "http://172.20.46.51:7010/sj_assets/v6/api/ecomn/raster_result"
    
    print("\n" + "=" * 50)
    print("测试多个省份编码")
    print("=" * 50)
    
    for test_case in test_cases:
        params = {
            'ak': 'mf85056077e36b72b8ef9170acd8d95b9e',
            'op': 'select_with_ref',
            'page_count': 200000,
            'page_num': 1,
            'file_type': 'gdb',
            'release_status': 'online',
            'usage_status': 'ready',
            'admin_province_code': test_case['admin_province_code'],
            'release_year': test_case['release_year']
        }
        
        print(f"\n测试: {test_case['description']}")
        print(f"参数: admin_province_code={test_case['admin_province_code']}, release_year={test_case['release_year']}")
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'item_list' in data['result']:
                    count = len(data['result']['item_list'])
                    if count > 0:
                        file_path = data['result']['item_list'][0].get('file_path', 'N/A')
                        print(f"  ✓ 成功, 找到 {count} 个文件, 路径: {file_path}")
                    else:
                        print(f"  ⚠ 成功但无数据")
                else:
                    print(f"  ✗ 响应格式异常")
            else:
                print(f"  ✗ HTTP错误: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ 请求失败: {e}")


if __name__ == '__main__':
    # 安装依赖（如果尚未安装）
    try:
        import requests
    except ImportError:
        print("请先安装requests库: pip install requests")
        exit(1)
    
    # 运行测试
    #test_gdb_api()
    #test_multiple_provinces()

def get_gdb_path_from_api(admin_province_code, release_year):
    """通过API动态获取GDB文件路径"""
    try:
        # API参数
        params = {
            'ak': 'mf85056077e36b72b8ef9170acd8d95b9e',
            'op': 'select_with_ref',
            'page_count': 200000,
            'page_num': 1,
            'file_type': 'gdb',
            'release_status': 'online',
            'usage_status': 'ready',
            'admin_province_code': admin_province_code,
            'release_year': release_year
        }
        
        # API地址
        api_url = "http://172.20.46.51:7010/sj_assets/v6/api/ecomn/raster_result"
        # 发送请求
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(data) 
        # 检查返回结果
        if 'result' in data and 'item_list' in data['result'] and data['result']['item_list']:
            file_path = data['result']['item_list'][0]['file_path']
            return file_path
        else:
            print('error')
            return "./assets/2023年广东省分县现状.gdb"
            
    except Exception as e:
        print(e)
        return "./assets/2023年广东省分县现状.gdb"

print(get_gdb_path_from_api(510000,2024))
