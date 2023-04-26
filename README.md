# YOLOv3 
Most of the code is based on YOLOv3 paper, but small parts are modified.  

## Inference coco data(gt vs pred)  
Coco data is too huge, so i trained samll epoch.  
### YOLOv3 
mAP50: 0.10606894  
![truth_and_pred_0](https://user-images.githubusercontent.com/42567320/234242047-9445f6b7-3d15-4366-83eb-d27478093c67.jpg) 
![truth_and_pred_1](https://user-images.githubusercontent.com/42567320/234242061-15b216ea-a71a-4e6e-afa9-64af04297c48.jpg) 
![truth_and_pred_2](https://user-images.githubusercontent.com/42567320/234242074-29808aff-40af-4340-9578-87e968f93964.jpg) 
![truth_and_pred_3](https://user-images.githubusercontent.com/42567320/234242081-ae0f59e9-e602-4c44-b36b-2110e9091082.jpg) 

## Inference voc data(gt vs pred)  
I think mAP is too low, because voc data is labeled as loose standard.  
We can see loose label in gt_images.  
But prediction works well.
### YOLOv3  
mAP50: 0.25553376  
![truth_and_pred_0](https://user-images.githubusercontent.com/42567320/234242697-2c947fc4-c166-4a2b-bb71-8425e0f8558b.jpg) 
![truth_and_pred_3](https://user-images.githubusercontent.com/42567320/234242706-55a0ab23-6c52-4b30-997b-fb184ef8cf13.jpg) 
![truth_and_pred_4](https://user-images.githubusercontent.com/42567320/234242717-fed1e652-b946-4efe-b43d-ca5b998720af.jpg) 
![truth_and_pred_5](https://user-images.githubusercontent.com/42567320/234242726-a870bd3f-bce1-409f-a104-01384b0ebec0.jpg) 
### YOLOv3_tiny  
mAP50: 0.22488195   
![truth_and_pred_0](https://user-images.githubusercontent.com/42567320/234242889-6c62adb7-5b13-4267-8fe2-953584678a6d.jpg) 
![truth_and_pred_3](https://user-images.githubusercontent.com/42567320/234242898-a12f13ce-adb6-444c-8879-7c64c2a49718.jpg) 
![truth_and_pred_4](https://user-images.githubusercontent.com/42567320/234242908-874592c1-60f3-4de9-90c3-82e2ed0dc23e.jpg) 
![truth_and_pred_5](https://user-images.githubusercontent.com/42567320/234242922-f1ff0935-a06d-4450-a8b2-a93a80298465.jpg) 

## Inference custom data(gt vs pred)  
### YOLOv3  
mAP50: 0.75312650  
![truth_and_pred_0](https://user-images.githubusercontent.com/42567320/234499401-abb31f70-197d-40c6-af32-b401227f6bc8.jpg)
![truth_and_pred_1](https://user-images.githubusercontent.com/42567320/234499417-ed3bdc9a-f68e-4c29-bda4-c849e78fe533.jpg)
![truth_and_pred_2](https://user-images.githubusercontent.com/42567320/234499426-373f96ff-76a8-43c6-9e47-b82536a8a65d.jpg)
### YOLOv3_tiny
mAP50: 0.74103535
![truth_and_pred_0](https://user-images.githubusercontent.com/42567320/234499580-b379280d-3570-4235-9368-90a955f8ee00.jpg)
![truth_and_pred_1](https://user-images.githubusercontent.com/42567320/234499591-1f1e2c5e-b349-4229-96aa-d0ac31a84c36.jpg)
![truth_and_pred_2](https://user-images.githubusercontent.com/42567320/234499596-eb2f0496-ef7c-49c7-8055-6093d067241a.jpg)


## pretrained model 
https://drive.google.com/drive/folders/1AwPJ_N0durYlG2zpokQjVlg2CKyGfr7L?usp=sharing
