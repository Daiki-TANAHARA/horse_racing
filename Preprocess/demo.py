from preprocess import read_race_csv, preprocess_train_valid

features = [
    "レースID", "レース日付", "着順",
    "距離(m)", "斤量", "馬齢", "馬体重", "枠番", "馬番",
    "騎手", "調教師", "芝・ダート区分", "競馬場名", "天候", "馬場状態1",
    "past3_hukusho_rate", "past5_hukusho_rate", "past9_hukusho_rate",
    "past5_agari_mean", "past5_chakujun_mean", "career_count",
    "days_since_last_race", "prev_chakujun", "prev_agari", "prev_ninki",
    "past_dist_hukusho_rate", "past_track_hukusho_rate",
    "jockey_hukusho_rate", "trainer_hukusho_rate", "jockey_course_rate",
    "past5_hukusho_rate_race_z", "past5_agari_mean_race_z",
    "jockey_hukusho_rate_race_z", "past5_chakujun_mean_race_z",
    "馬体重_race_z",
]

# usecolsで絞って読み込む(メモリ対策として重要)
df = read_race_csv("../data/19860105-20210731_race_result.csv", features)

train_processed, valid_processed, preprocessor = preprocess_train_valid(
    df, features, split_date="2019-01-01"
)

print(train_processed.shape, valid_processed.shape)
# → (1488790, 53) (123106, 53)