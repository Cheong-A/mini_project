
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline 주피터 노트북에 옮길때만

sns.set(font_scale=2)
data = pd.read_csv('D:\\mini_project\\data\\train.csv')

print(data.head())
print(data.shape)

print(data.isnull().sum())
#직업 유형에서 8171개의 결측치가 있음

##Credit 비율 살펴보기
f, ax = plt.subplots(1,2, figsize=(18,8))
data['credit'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Credit')
ax[0].set_ylabel('credit')
sns.countplot('credit',data=data,ax=ax[1])
ax[1].set_title('credit')
plt.show()

#credit변수는 낮을수록 높은 신용도 -> 여기서는 낮은 신용의 신카 사용자가 많은 비율 차지


##데이터 타입 파악
#순서가 존재 하지 않는 Categorical Features
#Binary
#gender: 성별
#car: 차량 소유 여부
#reality: 부동산 소유 여부
#FLAG_MOBIL: 핸드폰 소유 여부
##work_phone: 업무용 전화 소유 여부
#phone: 전화 소유 여부
#email: 이메일 소유 여부
#Multi
#family_size: 가족 규모
#house_type: 생활 방식
#occyp_type: 직업 유형
#income_type: 소득 분류
#family_type: 결혼 여부

#순서가 존재하는 Ordinal Features
#edu_type: 교육 수준
#credit : 신용 기

###변수별 신용도 EDA
#성별
print(data.groupby(['gender'])['credit'].count())
print(data.groupby(['gender','credit'])['credit'].count())

f, ax = plt.subplots(1,2,figsize=(18,8))
data[['gender','credit']].groupby(['gender']).mean().plot.bar(ax=ax[0])
ax[0].set_title('credit vs gender')
sns.countplot('gender',hue='credit',data=data,ax=ax[1])
ax[1].set_title('gender:credit level')
plt.show()

#여자가 연체율이 높은 것처럼 보이나, 비율에서 차이가 없고, 여성의 데이터가 2배가량 많이 수집됨

pd.crosstab(data.credit,data.gender,margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab(data.gender,data.credit,margins=True).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(18,8))
data['gender'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of people By gender')
ax[0].set_ylabel('Count')
sns.countplot('credit',hue='gender',data=data,ax=ax[1])
ax[1].set_title('credit: male vs female')
plt.show()

##차량 소유 여부에 따라서 신용점수는?
print(data.groupby(['car'])['credit'].count())
print(data.groupby(['car', 'credit'])['credit'].count())

f,ax=plt.subplots(1,2,figsize=(18,8))
data[['reality','credit']].groupby(['reality']).mean().plot.bar(ax=ax[0])
ax[0].set_title('credit vs reality')
sns.countplot('reality',hue='credit',data=data,ax=ax[1])
ax[1].set_title('reality:credit level')
plt.show()

pd.crosstab(data.credit,data.reality,margins=True).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(18,8))
data['reality'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of people By reality')
ax[0].set_ylabel('Count')
sns.countplot('credit',hue='reality',data=data,ax=ax[1])
ax[1].set_title('credit: reality vs No reality')
plt.show()

#결혼 여부에 따라서 신용점수는 ?
print(data.groupby(['family_type'])['credit'].count())
print(data.groupby(['family_type','credit'])['credit'].count())

f,ax=plt.subplots(1,2,figsize=(18,8))
data[['family_type','credit']].groupby(['family_type']).mean().plot.bar(ax=ax[0])
ax[0].set_title('credit vs family_type')
sns.countplot('family_type',hue='credit',data=data,ax=ax[1])
ax[1].set_title('family_type:credit level')
plt.xticks(rotation=60)
plt.show()

pd.crosstab(data.credit,data.family_type,margins=True).style.background_gradient(cmap='summer_r')

f,ax=plt.subplots(1,2,figsize=(18,8))
data['credit'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number of people By credit')
ax[0].set_ylabel('Count')
sns.countplot('credit',hue='family_type',data=data,ax=ax[1])
ax[1].set_title('credit: family_type')
plt.show()

