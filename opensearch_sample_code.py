import tensorflow as tf
import tensorflow_hub as hub
import cv2


# cv2 type의 Image List
def cv2feature_extraction(cv_imgs: list):
    '''

    :param cv_imgs: 추출하고자 하는 cv2이미지의 리스트입니다.
    :return: feature_extraction이 완료된 ndarray입니다.
    '''
    module_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1'
    module = hub.load(module_url)
    width = height = 96

    x = [cv2.resize(img, (width, height)) / 255. for img in cv_imgs]
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x = module.signatures['default'](x)['default'].numpy()
    return x


# Key: 이미지 ID, Value:[feature_vector, byte_image]
def record2json(img_dict:dict):
    ###
    with open("{save_path}/{file_name}.json", "a", encoding="utf-8") as f:
        n = 0
        for c, r in img_dict.items():
            index_line = dict({"index": {"_index": "{index_name}", "_id": ""}})
            index_line['index']['_id'] = n
            n += 1
            h = json.dumps(index_line, ensure_ascii=False)
            f.write(h + "\n")

            resource_line = dict({"imageId": "", "feature": [], "byteImg": ""})
            resource_line["imageId"] = c
            resource_line["feature"] = r[0].tolist()
            resource_line["byteImg"] = str(base64.b64encode(r[1]).decode('utf-8'))

            e = json.dumps(resource_line, ensure_ascii=False)
            f.write(e + "\n")
            time.sleep(0.05)
        f.close()

