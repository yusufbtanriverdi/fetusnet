import pandas as pd

# Read the dataframe, preserving OSIDs as strings.
df = pd.read_csv("/media/yusuf/HDD 4TB/DATA_reproduced_fin/sinMaternitat__fold3.csv", dtype={"osid": "string"})

# Read the test OSIDs, ignoring empty lines and surrounding whitespace.
with open("temp/test_list_ga.txt", "r", encoding="utf-8-sig") as file:
        testListGa = file.read()
# Assign 1 if the OSID occurs in the file string, otherwise 0.
df["TEST_LIST_GA"] = df["osid"].astype(str).str.strip().apply(
    lambda osid: int(bool(osid) and osid in testListGa)
)

df.to_csv("/media/yusuf/HDD 4TB/DATA_reproduced_fin/3dfetus_test_fold3.csv")
print(df[df["TEST_LIST_GA"]==1])