# AICE_1

# 🧠 AICE ASSOCIATE 자격증 준비

이 프로젝트는 Pandas, Scikit-learn, TensorFlow 등을 사용하여 데이터 전처리, 시각화, 머신러닝 및 딥러닝 회귀 모델을 설계하고 성능을 평가하는 과제를 다룹니다.

---

## 📌 문제 목록 및 요구사항

### 3. 데이터 로딩 및 병합
- `4000TT.csv`, `input.csv` 파일을 Pandas로 불러옵니다.
- `machine_name`을 기준으로 병합합니다.
- `input` 컬럼만 추출하여 `infoDF`에 저장합니다.

### 4. 주소에서 행정동 추출 및 시각화
- `Address` 컬럼에서 행정동(마지막 단어)을 추출하여 새로운 컬럼 생성
- 행정동 빈도를 countplot으로 시각화 (x축: 행정동, y축: 빈도수)

### 5. 주행시간 및 평균속도 분포 시각화
- `Time_Driving`, `Speed_Per_Hour` 컬럼을 각각 시각화 (히스토그램 또는 KDE)
- Seaborn을 활용하여 그래프 출력

### 6. 이상치 처리
- `Speed_Per_Hour ≥ 300`인 이상치 식별
- 조건에 따라 해당 데이터를 제거하거나 최대값으로 조정
- np.where 또는 drop을 활용해 `infoDF`에 반영

### 7. 데이터 조건 필터링
- `Speed_Per_Hour`: 10 이상 300 이하
- `Time_Driving`: 1 이상 1000 이하
- 위 조건을 모두 만족하는 데이터만 추출하여 `infoDF` 갱신

### 8. 수치형 정규화
- `Speed_Per_Hour`, `Time_Driving` 컬럼에 대해 MinMax 또는 Standard 정규화 수행
- 정규화된 결과는 새로운 컬럼으로 추가 (`Speed_Scaled`, `Time_Scaled` 등)

### 9. 원-핫 인코딩
- `Address` 컬럼에서 추출된 행정동을 기준으로 one-hot encoding 수행
- `pd.get_dummies()`를 활용하고 결과를 `infoDF`에 반영

### 10. 훈련/검증 데이터 분리 및 스케일링
- `Time_Driving` → y (label), 나머지 컬럼 → X (feature)
- train_test_split (80:20, random_state=42)로 분할
- `StandardScaler`를 적용하여 `X_train`, `X_test` 스케일링
- 스케일된 데이터를 `DataScaled`로 저장

### 11. 머신러닝 모델 (회귀)
- `DecisionTreeRegressor`, `RandomForestRegressor` 모델 학습
- 타겟: `Time_Driving`
- RMSE(Root Mean Squared Error)를 지표로 성능 평가
- 예측 결과(`y_test`, `y_pred`)를 시각화하여 실제값과 비교

### 12. MAE 기반 모델 비교
- `mean_absolute_error`를 사용하여 MAE 평가
- 각 모델의 MAE 출력 및 비교
- 어떤 모델이 더 우수한지 해석 포함

### 13. 딥러닝 모델 설계 및 학습
- TensorFlow(Keras)로 Input → Hidden → Output 구조 모델 구성
- 손실 함수: `mae`, 지표: `mae`, `mse`, optimizer: `adam`
- `fit()`으로 학습 후, `history`에 기록 저장
- 학습 결과(`loss`, `mae`) 시각화

### 14. 딥러닝 성능 평가 (MSE 시각화)
- `history`에서 `mse`, `val_mse`를 꺼내 line chart로 시각화
- x축: Epochs, y축: MSE
- 학습 MSE와 검증 MSE를 두 선으로 표현
