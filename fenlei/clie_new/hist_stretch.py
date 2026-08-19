import numpy as np
import cv2
import numbers
import time
import traceback

def GetMaskArea(I0,rad = None,isExpand = False,min_valid_value = 0,max_valid_value = None):
    
    if I0.min() > min_valid_value:
        return None
    else:
        if rad is None:
            rad = 7
        #kernel = np.ones((15,15),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(rad),int(rad)))
        iter_number = 2
        edge_mask_min_value = 10
        edge_width = 30
        data_edge = np.ones_like(I0)
        data_edge[edge_width:-edge_width,edge_width:-edge_width] = 0
        edge_mask = np.logical_and(I0<edge_mask_min_value,data_edge)     
        mask_array = (I0 <= min_valid_value)
        mask_array = np.logical_or(mask_array,edge_mask)#.astype(np.uint8)
        if max_valid_value is not None:
            max_mask = (I0 >= max_valid_value)
            mask_array = np.logical_or(mask_array,max_mask)
        #if I0_mean_count >= 3:
        #    mask_array = I0 == I0_min
        #if I0_max_count >=3:
        #    mask_array = (mask_array == 1) | ( I0 == I0_max)     
        mask_array = mask_array.astype(np.uint8)
        #print('mask_array:',mask_array.mean())
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_OPEN, kernel,iter_number)
        mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel,iter_number)
        #kernel = np.ones((3,3),np.uint8)
        if isExpand:
            mask_array = cv2.morphologyEx(mask_array, cv2.MORPH_DILATE, kernel,iter_number)
        #print('mask_array:',mask_array.mean())
        mask_array = mask_array.astype(np.bool_)
        if mask_array.any():
            return mask_array
        else:
 
            return None
def CalHistogram(img,left_mask = None, right_mask = None,is_resample = True,nodata = None,
                left_clip = None,right_clip = None,is_clip = False):
    
    img_dtype = img.dtype 

    img_0 = img.reshape(-1)
    col_raw = img_0.shape[0]
    col_hist = 2048 * 2048 * 4
    down_scale = (col_raw / col_hist)

    #print('down scale:',down_scale)
    img_min,img_max = img.min(),img.max()

    if down_scale > 1:
        V00 = np.arange(round(img_0.shape[0]/down_scale))    
        V00 = (V00 * down_scale).astype(np.uint32)
        img_hist = img_0[V00]#cv2.resize(img,(row_hist,col_hist),interpolation = cv2.INTER_NEAREST)
    else:
        img_hist = img_0

    if img_hist.min() != img_hist.max():
        img_hist = img_hist[img_hist != nodata]
    img_min,img_max = img_hist.min(),img_hist.max()
    n_bins = 2 ** 16
    if (img_dtype == np.uint8 ):
        n_bins = 256
    if  (img_dtype == np.uint16 ):
        n_bins = 2**16
    elif (img_dtype == np.uint32):
        n_bins = 2**32 

    if (img_dtype == np.uint8 ) or (img_dtype == np.uint16 ) or (img_dtype == np.uint32):
        #print(img_dtype)
        #s_values,hist = np.unique(img_hist[img_hist>0], return_counts=True)
        hist = np.bincount(img_hist,minlength = n_bins)      
        s_values = np.arange(n_bins)        
    else:
        hist,s_values= np.histogram(img_hist,bins = n_bins,range = (img_min,img_max))
    
    left_mask_clip = 0 if left_mask is None else left_mask
    right_mask_clip = -1 if right_mask is None else -right_mask          
    hist[:left_mask_clip] = 0
    hist[right_mask_clip:] = 0
    s_values = np.clip(s_values,s_values[left_mask_clip],s_values[right_mask_clip])    
    return hist,s_values
def clip_histogram(hist, clip_limit = None,left_valid = None,right_valid = None):
    """Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    hist = hist.astype(float)

    #if left_valid is not None:
    left_valid = 0 if left_valid is None else left_valid
    right_valid = len(hist) if right_valid is None else right_valid

    valid_size = right_valid - left_valid
    valid_mask = np.zeros(len(hist))
    valid_mask[left_valid:right_valid] = 1
    valid_mask = valid_mask.astype(np.bool_)


    if clip_limit is None:
        clip_limit = 0.02

    hist_sum = hist.sum()
    clip_limit = clip_limit * hist_sum

    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // valid_size #hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = np.logical_and(hist < upper , valid_mask)
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit
    #print(hist.size)
    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(left_valid,right_valid):#hist.size):

            try:
                under_mask = np.logical_and(hist < clip_limit,valid_mask)
                step_size = int(max(1, np.count_nonzero(under_mask) // n_excess))
                under_mask = under_mask[index::step_size]
                hist[index::step_size][under_mask] += 1
                n_excess -= np.count_nonzero(under_mask)
                if n_excess <= 0:
                    break
            #except:
            except Exception as ex:
                print("have exception!")
                #print('can not dehaze file:',in_file)
                #continue
                traceback.print_exc()
                #continue                
                print(index,step_size)
                break
        if prev_n_excess == n_excess:
            break

    return hist

def GetPercentStretchValue(img,img_valid = None,left_clip = 0.001,right_clip = 0.001,left_mask = None,
    right_mask = None,is_resample = True,nodata = None,return_hist = False):
    
    left_clip_valid  = 1.0E-6#left_clip/100
    right_clip_valid = 1.0E-6#left_clip/100

    left_clip  = max(left_clip,left_clip_valid)
    right_clip = max(right_clip,right_clip_valid)

    left_clip = np.clip(left_clip,0,1)
    left_clip_valid = np.clip(left_clip_valid,0,1)
    right_clip = np.clip(right_clip,0,1)
    right_clip_valid = np.clip(right_clip_valid,0,1)

    right_clip = 1.0 - right_clip#0.98
    right_clip_valid = 1.0 - right_clip_valid    

    img_min = img.min()
    img_max = img.max()
    img_min_valid = img_min
    img_max_valid = img_max

    hist,s_values = CalHistogram(img,left_mask = left_mask,right_mask = right_mask,nodata = nodata)

    #CalHistogram(img,left_mask = None, right_mask = None,is_resample = True,nodata = None)      
    s_quantiles = np.cumsum(hist).astype(np.float64)
    #print(s_quantiles)
    s_quantiles /= (s_quantiles[-1] + 1.0E-5)

    left_clip_valid_index = np.argmin(np.abs(s_quantiles-left_clip_valid))
    left_clip_index = np.argmin(np.abs(s_quantiles-left_clip))
    right_clip_valid_index = np.argmin(np.abs(s_quantiles-right_clip_valid))
    right_clip_index = np.argmin(np.abs(s_quantiles-right_clip))  

    img_min_valid,img_min_clip,img_max_clip,img_max_valid = \
        s_values[[left_clip_valid_index,left_clip_index,right_clip_index,right_clip_valid_index]]
    img_min_valid = max(img_min_valid,img_min) 
    img_min_clip = max(img_min_clip,img_min_valid)
    img_max_valid = min(img_max,img_max_valid)
    img_max_clip = min(img_max_clip,img_max)
    #print('[img_min,img_min_valid,img_min_clip,img_max_clip,img_max_valid,img_max]')
    #print([img_min,img_min_valid,img_min_clip,img_max_clip,img_max_valid,img_max])
    if return_hist:
        return [img_min,img_min_valid,img_min_clip,img_max_clip,img_max_valid,img_max],hist,s_values 
    else:
        return [img_min,img_min_valid,img_min_clip,img_max_clip,img_max_valid,img_max]

def hist_equal_lut(input_image_data = None,hist_in = None,left_clip = None,right_clip = None,to_8bit = True,min_value = 0):
    
    if hist_in is  None:
        hist,_ = CalHistogram(input_image_data)#,left_clip = left_clip,right_clip = right_clip,is_clip = True) 
        percent_clip,hist,s_values = GetPercentStretchValue(input_image_data,return_hist=True) 
        left_valid = percent_clip[1]
        right_valid = percent_clip[4]      
        #print(len(hist))
        #hist = hist[:input_image_data.max()+2]
        #print(len(hist))
    else:
        hist = hist_in.copy()
        if (left_clip is not None and right_clip is not None):
            hist = hist[left_clip:right_clip]
        left_valid = 0
        right_valid = len(hist)           

    hist =clip_histogram(hist,left_valid = left_valid,right_valid = right_valid)

    #print(len(hist))
    #for i,l in enumerate(hist):
    #    print(i,'o'*8,l)        
    hist = np.cumsum(hist)
    hist = hist/(hist[-1]+0.01)

    #for i,l in enumerate(hist):
    #    print(i,'%'*8,l) 

    if to_8bit:
        scale = 256
    else:
        scale = 2 * 16
    #lut = np.arange(len(hist))/len(hist)
    #lut = np.interp(lut,hist,lut)
    lut = hist

    #for i,l in enumerate(lut):
    #    print(i,'*'*8,l)    

    lut = ((scale-min_value)*lut + min_value).astype(np.uint16)

    #for i,l in enumerate(lut):
    #    print(i,'='*8,l)
    #lut = ((len(lut)-min_value)/len(lut) * lut + min_value).astype(np.uint16) 
    return lut   

def percent_stretch_lut(input_image_data, left_clip = 0.001,right_clip = 0.001,left_mask = None,
                        right_mask = None,is_test = False,is_simple_stretch = False,to_8bit = False,
                        auto_stretch = False,gamma_stretch_value = None,left_min_value = 1):
    
    #input_image_data_raw = input_image_data.copy()

    start_time = time.perf_counter() 
    if input_image_data is None:
        return None
    
    if auto_stretch:
        left_clip = 0.0025
        right_clip = 0.001
  
    #if is_test:
    #    print(input_image_data.shape)    
    indtype = input_image_data.dtype
    if indtype == np.uint8:
        to_8bit = True
     
    img_clip_value_raw,hist,s_values = GetPercentStretchValue(input_image_data,left_clip=left_clip,
                right_clip = right_clip,left_mask = left_mask,right_mask=right_mask,return_hist=True)    
    if img_clip_value_raw is None:
        return None
    if is_test:
        print('min_value,min_valid,min_clip,max_clip,max_valid,max_value')
        print(img_clip_value_raw)
    if auto_stretch :
        a = img_clip_value_raw[2]
        b = img_clip_value_raw[3]
        c = a - 0.1 * (b-a)
        d = b + 0.5 * (b-a)
        img_clip_value_raw[1] = int(np.max((img_clip_value_raw[1],c)))
        img_clip_value_raw[4] = int(np.min((img_clip_value_raw[4],d)))     
        if is_test:
            print('---------adjusted...---------')
            print(img_clip_value_raw)      
            print('----')
    
    img_clip_value = img_clip_value_raw/(img_clip_value_raw[-1]+0.1)    
    min_value = img_clip_value[0]
    min_valid = img_clip_value[1]
    min_clip = img_clip_value[2]  #a
    max_clip = img_clip_value[3]  #b
    max_valid = img_clip_value[4]
    max_value = img_clip_value[-1]     
    
    dark_scale0  =  (min_valid -min_value)
    left_scale0  = (min_clip - min_valid) 
    middle_scale0 = (max_clip - min_clip)  
    right_scale0 = (max_valid - max_clip) 
    white_scale0 = (max_value - max_valid)

    if is_test:
        print([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0]) 
        print(np.array([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0])*255) 
        #print([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0]*255) 
    scale = np.sqrt(4)    
    dark_scale0   = np.clip( dark_scale0    * 1 / scale/2,0,10.0/255)#left_clip * 1)   ##10   
    left_scale0   = np.clip( left_scale0    * 1 / scale, 10/255, 60.0/255.0 )            ##60
    middle_scale0 = np.clip( middle_scale0  * scale,  100/255.0, 160.0/255.0 )           ##160 
    right_scale0  = np.clip( right_scale0   * 1 / scale, 10/255, 60.0/255.0 )            ##60
    white_scale0  = np.clip(white_scale0    * 1 / scale/2,0,10.0/255)#right_clip * 1)  ##10        

    total_scale = dark_scale0 + left_scale0 + middle_scale0 + right_scale0 + white_scale0

    dark_scale = dark_scale0 / total_scale
    left_scale   = left_scale0 / total_scale# (left_scale0 + middle_scale0 + right_scale0)   
    middle_scale = middle_scale0 / total_scale#(left_scale0 + middle_scale0 + right_scale0)
    right_scale  = right_scale0 / total_scale#(left_scale0 + middle_scale0 + right_scale0)
    white_scale = white_scale0 / total_scale           

    if is_test:
        print('cliped...')
        print(np.array([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0])) 
        print(np.array([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0])*255) 
        #print([dark_scale0,left_scale0,middle_scale0,right_scale0,white_scale0]*255)
        print('normalized...')
        print(np.array([dark_scale,left_scale,middle_scale,right_scale,white_scale])) 
        print(np.array([dark_scale,left_scale,middle_scale,right_scale,white_scale])*255) 
        #print([dark_scale,left_scale,middle_scale,right_scale,white_scale])             
    #stretch_scale = 2**8 - 1
    stretch_scale = img_clip_value_raw[-1]
    if input_image_data.dtype == np.uint8:
        lut_values = np.arange(2**8,dtype=np.uint8)
        stretch_scale = 2**8# - 1
    elif input_image_data.dtype == np.uint16:
        lut_values = np.arange(2**16,dtype=np.uint16)
        #stretch_scale = 2**16 - 1
    elif input_image_data.dtype == np.uint32:
        lut_values = np.arange(2**32,dtype=np.uint32)
        #stretch_scale = 2**32 - 1
    elif input_image_data.dtype == np.int16:
        lut_values = np.arange(2**16,dtype=np.int16)
    else:
        lut_values = np.arange(2**16,dtype=np.int16)

    if to_8bit:
        stretch_scale = 2 ** 8 

    stretch_scale_t = stretch_scale - left_min_value
    if is_test:
        print('stretch_scale_t:',stretch_scale_t)
    min_raw = img_clip_value_raw[0]
    min_valid_raw = img_clip_value_raw[1]
    min_clip_raw = img_clip_value_raw[2]
    max_clip_raw = img_clip_value_raw[3]
    max_valid_raw = img_clip_value_raw[4]
    max_raw = img_clip_value_raw[5]
    
    min_raw = max(1,min_raw)
    min_valid_raw = max(min_valid_raw,min_raw + 1)
    min_clip_raw = max(min_clip_raw,min_valid_raw + 1)
    max_valid_raw = min(max_valid_raw,max_raw-1)
    max_clip_raw = min(max_clip_raw,max_valid_raw-1) 

    (min_raw,min_valid_raw,min_clip_raw,max_clip_raw,max_valid_raw,max_raw) = [int(a) for a in [min_raw,min_valid_raw,min_clip_raw,max_clip_raw,max_valid_raw,max_raw]]

    #print(img_clip_value_raw) 
    #stretch_scale_t = stretch_scale - left_min_value
    lut_dtype = lut_values.dtype
    lut_values = lut_values.astype(np.float32)
    lut_values = np.clip(lut_values,0,max_raw)
    if min_raw > 0:
        lut_values[:min_raw] = left_min_value
    lut_values[min_raw:min_valid_raw]      = np.ceil((lut_values[min_raw:min_valid_raw]       - min_raw)/(min_valid_raw - min_raw + 1)            * dark_scale   * stretch_scale_t + lut_values[min_raw-1])#+ left_min_value)   
    lut_values[min_valid_raw:min_clip_raw] = np.ceil((lut_values[min_valid_raw:min_clip_raw]  - min_valid_raw)/(min_clip_raw - min_valid_raw + 1) * left_scale   * stretch_scale_t + lut_values[min_valid_raw-1])#dark_scale   * stretch_scale_t + left_min_value)
    lut_values[min_clip_raw:max_clip_raw]  = np.ceil(( lut_values[min_clip_raw:max_clip_raw]  - min_clip_raw)/(max_clip_raw - min_clip_raw + 1)   * middle_scale * stretch_scale_t + lut_values[min_clip_raw-1])#left_scale   * stretch_scale_t + dark_scale   * stretch_scale_t + left_min_value)
    
    lut_values[max_clip_raw:max_raw] = hist_equal_lut(hist_in=hist,left_clip=max_clip_raw,right_clip=max_raw,to_8bit=True,min_value = lut_values[max_clip_raw-1] )
    #lut_values[max_clip_raw:max_valid_raw] = np.ceil(( lut_values[max_clip_raw:max_valid_raw] - max_clip_raw)/(max_valid_raw - max_clip_raw + 1)  * right_scale  * stretch_scale_t + lut_values[max_clip_raw-1])#middle_scale * stretch_scale_t + left_scale   * stretch_scale_t + dark_scale * stretch_scale_t + left_min_value)
    #lut_values[max_valid_raw:]             = np.ceil(( lut_values[max_valid_raw:]           - max_valid_raw)/(max_raw - max_valid_raw + 1)      * white_scale  * stretch_scale_t + lut_values[max_valid_raw-1])#right_scale  * stretch_scale_t + middle_scale * stretch_scale_t + left_scale * stretch_scale_t + dark_scale * stretch_scale_t + left_min_value)
    lut_values = np.clip(lut_values,left_min_value,stretch_scale-1).astype(lut_dtype)
    #lut_values = np.clip(lut_values,left_min_value,max_raw).astype(lut_dtype) 

    if is_test and False:
        #print('stretch_scale:',stretch_scale)
        with open('D:\\' + str(time.perf_counter()) + '00_tst_lut.csv', 'w+') as f:
            for i,lut_value in enumerate(lut_values):                
                f.write(' '.join([str(i),'---',str(lut_value),'\n']))
            f.close()

    
    if gamma_stretch_value is not None:
        lut_values = np.round(np.power(lut_values/lut_values.max(),gamma_stretch_value) * lut_values.max())
        lut_values = lut_values.astype(lut_dtype)

    end_time = time.perf_counter()
    time_interval = end_time - start_time
    start_time = time.perf_counter() 
    if is_test:
        print('histogram stretch lut time:',time_interval) 
    return lut_values 

def percent_stretch_image(input_image_data,left_clip = 0.001,right_clip = 0.001,left_mask = None,
                        right_mask = None,is_test = False,is_simple_stretch = False,is_return_mask = False,
                        to_8bit = False,auto_stretch = False,gamma_stretch_value = None,left_min_value=1,
                        nodata_value = 0 ):
    
    #input_image_data_raw = input_image_data.copy()
    if input_image_data is None:
        return None
    
    if auto_stretch:
        left_clip = 0.0025
        right_clip = 0.001

    start_time = time.perf_counter() 
    end_time = time.perf_counter()
    time_interval = end_time - start_time
    start_time = time.perf_counter() 
    n_dim = input_image_data.ndim
    img_bands = 1 if n_dim == 2 else input_image_data.shape[n_dim-1]

    if is_test:
        print(input_image_data.shape)

    xsize = input_image_data.shape[1]
    ysize = input_image_data.shape[0]

    if is_test:
        print('img_bands:',img_bands)    
    indtype = input_image_data.dtype

    if indtype == np.uint8:
        to_8bit = True

    if img_bands > 1:
        if to_8bit:
            out_8bit_data = np.zeros((ysize,xsize,img_bands),dtype = np.uint8)#+255
        else:
            out_8bit_data = np.zeros((ysize,xsize,img_bands),dtype = np.uint16)
    else:
        if to_8bit:
            out_8bit_data = np.zeros((ysize,xsize),dtype = np.uint8)#+255
        else:
            out_8bit_data = np.zeros((ysize,xsize),dtype = np.uint16)       
  
    if img_bands == 1:
        #image_mask = input_image_data == 0
        image_mask = GetMaskArea(input_image_data,min_valid_value= nodata_value)
        #if not image_mask.any():
        #    image_mask = None
    else:# img_bands == 1:
        image_mask = np.max(input_image_data,axis=2)# == 0
        image_mask = GetMaskArea(image_mask,min_valid_value= nodata_value)
        #if not image_mask.any():
        #    image_mask = None        
    
    end_time = time.perf_counter()
    time_interval = end_time - start_time
    start_time = time.perf_counter() 
    if is_test:
        print('create mask data time:',time_interval)       
    
    for i_band in range(img_bands):

        if img_bands == 1:
            input_image_data_raw = input_image_data#[:,:,i_band]
        else:
            input_image_data_raw = input_image_data[:,:,i_band]
        
        input_image_data_tmp = input_image_data_raw#.copy()

        if (input_image_data_tmp.dtype == np.float32) or (input_image_data_tmp.dtype == np.float64):
            input_image_data_tmp = (input_image_data_tmp - input_image_data_tmp.min())/(input_image_data_tmp.max() - input_image_data_tmp.min()) * (2 ** 16 -1)
            input_image_data_tmp = input_image_data_tmp.astype(np.uint16)               

        lut_values = percent_stretch_lut(input_image_data_tmp, left_clip = left_clip,right_clip = right_clip,left_mask = 1,
                        right_mask = 1,is_test = is_test,is_simple_stretch = False,to_8bit = to_8bit,
                        auto_stretch = auto_stretch,left_min_value = 1)


        #for i,lut_value in enumerate(lut_values):
        #    print(i,'---',lut_value)
        lut_dtype = lut_values.dtype        
        if gamma_stretch_value is not None:
            lut_values = np.round(np.power(lut_values/lut_values.max(),gamma_stretch_value) * lut_values.max())
            lut_values = lut_values.astype(lut_dtype)

        input_image_data_tmp = lut_values[input_image_data_tmp]#np.clip(input_image_data,0,2**8-1)

        if img_bands > 1:
            detail_scale = 1.6
        else:
            detail_scale = 1.6
     

        end_time = time.perf_counter()
        time_interval = end_time - start_time
        start_time = time.perf_counter() 
        if is_test:
            print('histogram stretch time:',time_interval)  
        if image_mask is not None:
            #print('image_mask is not none...')
            input_image_data_tmp[image_mask] = 0

        if img_bands > 1:
            #print(out_8bit_data[:,:,i_band].shape,input_image_data.shape)
            out_8bit_data[:,:,i_band] = input_image_data_tmp
        else:
            out_8bit_data = input_image_data_tmp
        if is_test:
            print('percent stretch image process over...')
    if is_return_mask:
        if image_mask is not None:
            return out_8bit_data,image_mask
        else:
            return out_8bit_data,None
    else:
        return out_8bit_data#,out_8bit_raw