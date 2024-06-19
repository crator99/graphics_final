# Graphics Final

## 실행 환경

- Python 3.11
- OpenCV 4.10
- Numpy 1.23.5

## 실행 방법
``` 
python imagesegmentation.py images/carla2.jpg  (다른 사진으로 원할 경우, jpg 파일 이름은 변경 가능)
```

"Select ROI"라는 이름의 창이 뜨면, ROI(Region of Interest)를 설정해주어야 함

top_left, top_right, bottom_right, bottom_left 순서로 점을 지정해준 뒤, 스페이스바

"Planting OBJ seeds"라는 이름의 창이 뜨면, 객체를 색칠해준 뒤 esc

"Planting BKG seeds"라는 이름의 창이 뜨면, 배경을 색칠해준 뒤 esc

계산을 거쳐 객체 테두리를 띄움

## Examples

1. `images/carla1.jpg` 

원본

![원본](https://github.com/crator99/graphics_final/blob/main/images/carla1.jpg)

라벨링

![라벨링](https://github.com/crator99/graphics_final/blob/main/images/carla1seeded.jpg)

객체 테두리 식별

![객체 테두리 식별](https://github.com/crator99/graphics_final/blob/main/images/carla1cut.jpg)

2. `images/carla2.jpg`

원본

![원본](https://github.com/crator99/graphics_final/blob/main/images/carla2.jpg)

라벨링

![라벨링](https://github.com/crator99/graphics_final/blob/main/images/carla2seeded.jpg)

객체 테두리 식별

![객체 테두리 식별](https://github.com/crator99/graphics_final/blob/main/images/carla2cut.jpg)


