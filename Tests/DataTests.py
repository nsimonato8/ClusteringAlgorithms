from datetime import datetime

# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Data Validation
# session = pd.read_csv("../TestData/sessions.csv")
#
# print(f"Before:\n{session.dtypes}\nSize: {session.size}\n")
#
# session.drop(["action_detail"], axis=1, inplace=True)
# session.dropna(inplace=True)
# encoder = BaseNEncoder(cols=["action", "action_type", "device_type"], return_df=True)
# session = encoder.fit_transform(session)
# session['user_id'] = session['user_id'].apply(lambda x: int(x, 36))
# session.to_csv("../TestData/session_cleaned.csv", sep=",")
# session = session.sample(frac=0.005, random_state=11)
# session.to_csv("../TestData/session_sample.csv", sep=",")
#
#
# print(f"After:\n{session.dtypes}\nSize: {session.size}\n")

session = pd.read_csv("../TestData/session_sample.csv")

print(f"{session.dtypes}")

model = [None] * 150
score = [None] * 150
k = range(10000, 10010)
for i in k:
    print(f"Test {i - k[0] + 1}. n_clusters={i}")
    timestamp = datetime.now()
    model[i - k[0]] = KMeans(n_clusters=i + 1, init='random', max_iter=200, n_init=10, random_state=1)
    model[i - k[0]].fit(session)
    score[i - k[0]] = model[i - k[0]].score(session)
    timestamp = datetime.now() - timestamp
    print(f"Time elapsed: {int(round(timestamp.total_seconds()))} seconds. Score: {score[i - k[0]]}")

# plt.plot(k, score)
