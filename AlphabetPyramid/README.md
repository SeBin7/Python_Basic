# Alphabet Pyramid

## 문제
```
1. 사용자로부터 문자열을 입력받고 문자열 내에서 같은 문자가 연속해서 등장하는 가장 긴 시퀀스를 찾아내어라.
2. 그 문자의 개수(max_count)를 기반으로 피라미드 형태를 출력하라. 피라미드는 max_count줄로 구성되며, 아래로 갈수록 문자의 개수가 증가한다.
```

**예시:**

입력:
```
Enter a string: aaabbccccdde
```
출력:
```
c, 4

   c
  ccc
 ccccc
ccccccc
```


## 셋업
먼저 다음 명령어로 가상환경을 만들고 활성화 시켜주자.
```sh
python3 -m venv venv
source venv/bin/activate
```

이후 필요한 패키지들을 설치해준다.
```sh
pip install -r requirements.txt
```

(optional) 만약 서비스 코드가 아니고 개발 코드라면 dev dependency 도 설치해준다.
```sh
pip install -r requirements-dev.txt
```

## 실행
다음 script 를 돌려서 코드를 실행할 수 있다.
```sh
python main.py
```

<br>
<br>

## (Optional) Test
test 코드가 작성되어 있을 때는 다음 명령어로 테스트를 진행할 수 있다.
```sh
python -m pytest
```

특정 test code 만 돌리고 싶다면 다음과 같이 쳐주자.
```sh
python -m pytest test_[your_test].py
```

## (Optional) Development

코드를 format 하고 싶으면 다음 명령어를 쳐보자.
```sh
black .
```

import 문의 순서를 보장하고 싶으면 다음 명령어를 치자.
```sh
isort .
```

코드의 typing 을 체크하고 싶으면 다음 명령어를 쳐보자.
```sh
mypy .
```