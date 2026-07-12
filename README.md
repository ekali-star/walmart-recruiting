# Walmart Recruiting - Store Sales Forecasting

## პროექტის მიმოხილვა

ეს არის ფინალური პროექტი Kaggle-ის კონკურსისთვის [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting). ამოცანა არის Time-Series პრობლემა — გვჭირდება 45 Walmart-ის მაღაზიის, თითოეულში მრავალი დეპარტამენტის, კვირეული გაყიდვების პროგნოზირება, გავითვალისწინეთ სეზონურობა, დღესასწაულები და markdown/ფასდაკლებები.

პროექტზე მუშაობდა 2 კაციანი გუნდი. მიზანი იყო რამდენიმე განსხვავებული არქიტექტურის მოდელის (Tree-Based, Classical Statistical, Deep Learning) გამოკვლევა, იმპლემენტაცია და შედარება.

## მონაცემები

გამოყენებული იქნა სამი ფაილი:
- **train.csv** — ისტორიული კვირეული გაყიდვები (Store, Dept, Date, Weekly_Sales, IsHoliday) — ~421,570 მწკრივი.
- **features.csv** — დამატებითი ინფორმაცია: ტემპერატურა, საწვავის ფასი, 5 MarkDown სვეტი, CPI, უმუშევრობის დონე.
- **stores.csv** — მაღაზიის ტიპი (A/B/C) და ზომა.

### EDA-ს ძირითადი დასკვნები

- **MarkDown სვეტები** (MarkDown1–5) დიდწილად ცარიელია 2011 წლის ნოემბრამდე პერიოდისთვის — ეს არის ფასდაკლების პროგრამის დაწყების თარიღი, ამიტომ ეს არის ლეგიტიმური "აკლია" და არა შემთხვევითი დაკარგვა.
- **უარყოფითი გაყიდვები** გვხვდება მცირე რაოდენობის მწკრივში — ეს არის დაბრუნებები (returns), არ წავშალეთ, რადგან რეალურ ინფორმაციას შეიცავს.
- **სეზონურობა** — გაყიდვები მკვეთრად იზრდება დღესასწაულების პერიოდში (Thanksgiving, Christmas, Super Bowl), რაც ასევე ასახულია შეფასების მეტრიკაში (WMAE holiday კვირებს 5x წონას ანიჭებს).
- სხვადასხვა დეპარტამენტს (Dept) აქვს რადიკალურად განსხვავებული სეზონური ხასიათი (მაგ. სათამაშოების დეპარტამენტი მკვეთრად იზრდება შობის წინ).

## Feature Engineering

აშენდა საერთო ფუნქცია `src/features.py`-ში (`build_features`), რომელსაც იყენებს ყველა მოდელის notebook:

- MarkDown სვეტების missing მნიშვნელობები შეივსო 0-ით (მნიშვნელობა: არ იყო აქტიური ფასდაკლება).
- `Total_MarkDown` — ხუთივე MarkDown სვეტის ჯამი.
- CPI და Unemployment — forward-fill მაღაზიის მიხედვით.
- დროზე დაფუძნებული feature-ები: Year, Month, Week.
- Store Type (A/B/C) — one-hot encoding.
- **Lag features**: `Sales_Lag_1` (წინა კვირის გაყიდვები) და `Sales_Lag_52` (იგივე კვირა წინა წელს) — ეს აღმოჩნდა ყველაზე მნიშვნელოვანი feature-ები ტრენინგისას.

### Train/Validation split და შეფასების მეტრიკა

გამოყენებულია დროზე დაფუძნებული split (`src/data_split.py`) — ბოლო 10 კვირა გამოიყო ვალიდაციისთვის, არა შემთხვევითი k-fold, რადგან ეს არის time-series ამოცანა. შეფასების მეტრიკა არის **WMAE** (Weighted Mean Absolute Error), სადაც holiday კვირებს ენიჭება 5x წონა — ეს ემთხვევა Kaggle-ის ოფიციალურ შეფასების ფორმულას.

## მოდელები და შედეგები

| არქიტექტურა | საუკეთესო კონფიგურაცია | Val Score | შენიშვნა |
|---|---|---|---|
| **XGBoost** | n_estimators=500, max_depth=10, learning_rate=0.03 | **WMAE: 1328.40** | საუკეთესო შედეგი ყველა მოდელს შორის |
| **LightGBM** | n_estimators=800, max_depth=-1, learning_rate=0.02, num_leaves=60 | WMAE: 1353.91 | ძალიან ახლოს XGBoost-თან, ოდნავ ჩამორჩება |
| **ARIMA** | order=(2,1,2), ერთ Store/Dept სერიაზე | MAE: 3118.13 | არ ითვალისწინებს სეზონურობას |
| **SARIMA** | order=(1,1,1), seasonal_order=(1,1,1,52) | MAE: 1224.64 | სეზონური წევრის დამატებამ ~60%-ით გააუმჯობესა შედეგი |

*შენიშვნა: ARIMA/SARIMA-ს მეტრიკა (MAE) არ არის პირდაპირ შედარებადი XGBoost/LightGBM-ის (WMAE) მეტრიკასთან, რადგან ARIMA/SARIMA მუშაობს ერთ კონკრეტულ Store/Dept სერიაზე, ხოლო tree-based მოდელები — მთელ dataset-ზე ერთდროულად.*

### რატომ სჭირდება რთული მოდელები ARIMA/SARIMA-ს გარდა

ARIMA და SARIMA მუშაობენ **ერთ** დროით სერიაზე ერთდროულად და არ იღებენ ბუნებრივად exogenous feature-ებს (markdown-ები, ტემპერატურა, CPI) SARIMAX-ის გარეშე. ჩვენს ამოცანაში გვაქვს ათასობით პარალელური Store×Dept სერია — ცალკეული ARIMA მოდელის დატრენინგება თითოეულისთვის არაპრაქტიკულია. ხის-დაფუძნებული მოდელები (XGBoost, LightGBM) სწავლობენ საერთო პატერნებს ყველა სერიაზე ერთდროულად, feature-ების საშუალებით (Store, Dept, Size, Type და ა.შ.), რაც ხსნის მათ უპირატესობას ამ ამოცანაზე.

## MLflow ექსპერიმენტების სტრუქტურა

ყველა ექსპერიმენტი ლოგირებულია DagsHub-ზე განთავსებულ MLflow tracking server-ზე:

- **XGBoost_Training**: XGBoost_baseline → XGBoost_with_lags → XGBoost_HPO_1/2/3 → XGBoost_final_pipeline
- **LightGBM_Training**: LightGBM_baseline → LightGBM_HPO_1/2/3 → LightGBM_final_pipeline
- **ARIMA_SARIMA_Training**: ARIMA_baseline, SARIMA_baseline

## საუკეთესო მოდელი და Model Registry

**XGBoost** (WMAE 1328.40) რეგისტრირებულია როგორც `walmart_sales_best_model` MLflow Model Registry-ში. მოდელი შენახულია სრულ sklearn Pipeline-ად, რომელიც შეიცავს:
1. `FeatureBuilder` — raw მონაცემებზე ავტომატურად აწარმოებს ყველა feature engineering ნაბიჯს (merge, cleaning, lag features).
2. `XGBRegressor` — თავად მოდელი.

ეს Pipeline პირდაპირ მუშაობს დაუმუშავებელ (raw) test set-ზე — `model_inference.ipynb`-ში ჩატვირთულია რეგისტრიდან და გამოიყენება Kaggle submission-ის დასაგენერირებლად.

`walmart_lightgbm_model` ასევე რეგისტრირებულია ცალკე, შედარებისთვის.

## გამოწვევები

**Lag feature-ების გაჟონვა (leakage) inference-ის დროს**: პირველადმა Pipeline-ის იმპლემენტაციამ Kaggle-ის test set-ზე გაცილებით ცუდი შედეგი აჩვენა (Public WMAE ~18,363), ვიდრე ვალიდაციაზე (1328.40). მიზეზი აღმოჩნდა ის, რომ `Sales_Lag_1`/`Sales_Lag_52` feature-ები test set-ზე ყოველთვის ხდებოდა 0, რადგან Kaggle-ის test.csv-ს არ გააჩნია `Weekly_Sales` სვეტი (ეს არის სწორედ ის, რასაც ვწინასწარმეტყველებთ) — ხოლო მოდელი ტრენინგისას ეყრდნობოდა რეალურ lag მნიშვნელობებს. ეს არის კარგი მაგალითი იმისა, თუ რატომ არის მნიშვნელოვანი Pipeline-ის ტესტირება ნამდვილ raw test მონაცემებზე, და არა მხოლოდ ვალიდაციის სეტზე. გამოსწორება მიმდინარეობს — `history_df` გადაეცემა `FeatureBuilder`-ს, რათა test set-ის lag feature-ები გამოითვალოს ტრენინგის ისტორიული მონაცემებიდან.

## Kaggle შედეგი

*[განახლდება inference bug-ის გასწორების შემდეგ]*

## შემდეგი ნაბიჯები

- Deep Learning მოდელების იმპლემენტაცია (DLinear, N-BEATS, TFT/PatchTST)
- Lag feature-ების bug-ის საბოლოო გასწორება და Kaggle-ზე ხელახლა submission
- საბოლოო README-ის დასრულება Kaggle score-ით და DL მოდელების შედეგებით

## რეპოზიტორიის სტრუქტურა

```
notebooks/
  model_experiment_XGBoost.ipynb
  model_experiment_LightGBM.ipynb
  model_experiment_ARIMA_SARIMA.ipynb
  model_experiment_DLinear.ipynb        (მომზადების პროცესშია)
  model_experiment_NBEATS.ipynb         (მომზადების პროცესშია)
  model_inference.ipynb
  00_EDA_FeatureEngineering.ipynb
src/
  features.py       — საერთო feature engineering
  data_split.py      — დროზე დაფუძნებული split და WMAE ფუნქცია
  mlflow_setup.py     — MLflow/DagsHub tracking-ის კონფიგურაცია
data/                 — Kaggle-ის მონაცემები (gitignored)
README.md
```

## გამოყენება (Setup)

1. `git clone` რეპოზიტორია და `pip install -r requirements.txt`.
2. შექმენით `.env` ფაილი `.env.example`-ის მიხედვით, თქვენი DagsHub username და token-ით.
3. ჩამოტვირთეთ მონაცემები [Kaggle-ის Data გვერდიდან](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data) და მოათავსეთ `data/` საქაღალდეში.
4. გაუშვით notebooks თანმიმდევრობით: ჯერ `00_EDA_FeatureEngineering.ipynb`, შემდეგ ცალკეული მოდელის notebook-ები.