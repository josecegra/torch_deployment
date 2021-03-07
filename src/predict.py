import os
import requests
import argparse

class ModelAPI():

    def __init__(self,host,endpoint_name,port):
        self.url = f"http://{host}:{port}/{endpoint_name}"

    def is_active(self):
        try:
            requests.get(self.url).json() 
            active = True
        except:
            active = False
        return active

    def class_index_dict(self):
        encoding_dict = None
        if self.is_active():
            resp = requests.get(self.url).json()
            encoding_dict = resp['class_index_dict']
        return encoding_dict
        
    def predict(self,img_path_list,XAI = False):
        if not self.is_active():
            return None
        if XAI:
            headers = {'XAI':'true'}
        else:
            headers = {'XAI':'false'}

        if isinstance(img_path_list,str):
            img_path_list = [img_path_list]

        pred_dict = {}
        bs = 4
        batches = [img_path_list[x:x+bs] for x in range(0, len(img_path_list), bs)]
        for batch in batches:
            file_dict = {f'file{i}':open(path,'rb') for i,path in enumerate(batch)}
            resp = requests.post(self.url, headers = headers, files=file_dict).json() 
            pred_dict.update(resp)

        return pred_dict


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint_name', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--XAI', required=False)
    args = parser.parse_args()

    XAI = False
    if args.XAI:
        XAI = bool(args.XAI)

    img_path_list = [os.path.join(args.img_dir,fname) for fname in os.listdir(args.img_dir)]

    api = ModelAPI(args.host,args.endpoint_name,int(args.port))
    if api.is_active():
        pred_dict = api.predict(img_path_list,XAI=XAI)
        print(pred_dict)
    else:
        print('flask api not running, check again!')

    


