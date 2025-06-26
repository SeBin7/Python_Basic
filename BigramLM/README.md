# Bi-gram Language Model


## 문제 상황
**1. 바이 그램 모델(Bi-gram Model) 구축**
* 문장을 구두점(., !, ?) 기준으로 분할 후 각 문장을 소문자로 변환
* 단어들을 토큰화한 후, 두 단어씩 짝지어 등장 빈도수를 딕셔너리 형태로 저장
    * 예: "i am a student" → ("i", "am"), ("am", "a"), ("a", "student")

**2. 예측**
* 사용자가 입력한 단어가 딕셔너리에 존재하면:
    * 그 단어 뒤에 등장했던 단어 중 가장 많이 등장한 단어를 예측 결과로 출력
    * 등장 확률도 계산하여 함께 출력
* 존재하지 않으면 "None" 출력



## 셋업

`BigramLM` 디렉토리 아래 `corpus.txt` 파일을 만들어주자. 이후, 관심있는 영어 뉴스 기사를 찾아서 모두 복사 붙여넣기 해서 `corpus.txt` 파일을 채워주자.

이후 가상환경을 설치해준 후

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

다음 파일을 돌려보자.

```sh
python original.py`
```
