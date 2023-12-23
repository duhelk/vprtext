
from PIL.ExifTags import TAGS
from PIL import Image
import piexif


def get_image_GPSInfo(image_path):
    try:
        image = Image.open(image_path)
        exifdata = image._getexif()
        for (tag_id,value) in exifdata.items():
            if TAGS.get(tag_id, tag_id) == 'GPSInfo':
                return value
    except:
        print('There was an error retrieving GPSInfo for', image_path)


def get_location_exif(image_file):
    exif_dict = piexif.load(image_file)
    for k, v in exif_dict['Exif'].items():
        if k == 37510:
            usercomment = v.decode("utf-8") 
            sensor_data = dict()
            for data in usercomment.split(']'):
                if data != '':
                    key,val = data.split("[")
                    sensor_data[key] = val
            lat, lon = sensor_data['LOCT'].split(',')
            return lat,lon 


##-- Read location from iphone images ---##
def get_image_location(image_path, pos_err=False):
    try:
        gps_info = get_image_GPSInfo(image_path)
        sign = 1 if gps_info[1] == 'N' else -1
        deg, min, sec = gps_info[2]
        deg = deg[0]/deg[1] if isinstance(deg, tuple) and len(deg) == 2 else deg
        min = min[0]/min[1] if isinstance(min, tuple) and len(min) == 2 else min
        sec = sec[0]/sec[1] if isinstance(sec, tuple) and len(sec) == 2 else sec
        lat = sign*(float(deg) + float(min)/60.0 + float(sec)/3600)

        sign = 1 if gps_info[3] == 'E' else -1
        deg, min, sec = gps_info[4]
        deg = deg[0]/deg[1] if isinstance(deg, tuple) and len(deg) == 2 else deg
        min = min[0]/min[1] if isinstance(min, tuple) and len(min) == 2 else min
        sec = sec[0]/sec[1] if isinstance(sec, tuple) and len(sec) == 2 else sec
        lon = sign*(float(deg) + float(min)/60.0 + float(sec)/3600)
        if pos_err:
            err = gps_info[31]
            err = err[0]/err[1] if isinstance(err, tuple) and len(err) == 2 else err
            return lat,lon, float(err)
        return lat,lon
    except:
        try:
            return get_location_exif(image_path)
        except:
            print('There was an error retrieving location info for', image_path)
