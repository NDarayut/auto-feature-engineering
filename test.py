from afe.benchmark import compare, BaselineMethod

def openfe(X_train, y_train, X_test, task):
    from openfe import OpenFE, transform
    feats = OpenFE().fit(data=X_train, label=y_train, task=task, n_jobs=1, verbose=False)
    return transform(X_train, X_test, feats[:10], n_jobs=1)

result = compare(
    methods=[BaselineMethod, openfe],
    datasets=["german-credit", "concrete-strength"],   # built-in suite keys
)
print(result)          # per-model-family markdown score table
df = result.to_frame() # raw rows as a pandas DataFrame