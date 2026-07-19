# Cross-domain class mapping

Kết quả overlap-class và full-set phải báo riêng. Không gộp một-nhiều class mà không ghi rõ.

## ExDark → VisDrone

| ExDark | VisDrone target | Ghi chú |
|---|---|---|
| Bicycle | bicycle (2) | Direct overlap |
| Bus | bus (8) | Direct overlap |
| Car | car (3) | Direct overlap |
| Motorbike | motor (9) | Name normalization |
| People | people (1) | Không đồng thời map sang pedestrian để tránh nhân đôi GT |

Boat, Bottle, Cat, Chair, Cup, Dog và Table chỉ nằm trong báo cáo full-set/excluded-class, không dùng trong overlap mAP của detector VisDrone.

## AU-AIR → VisDrone

| AU-AIR | VisDrone target | Ghi chú |
|---|---|---|
| person | people (1) | Chọn một target duy nhất |
| car | car (3) | Direct overlap |
| bus | bus (8) | Direct overlap |
| van | van (4) | Direct overlap |
| truck | truck (5) | Direct overlap |
| bike | bicycle (2) | Name normalization |
| motorbike | motor (9) | Name normalization |

Adjacent AU-AIR video frames không được chia ngẫu nhiên qua train/test. Evaluation cross-domain chỉ dùng split cố định theo sequence và không tune threshold trên test.
