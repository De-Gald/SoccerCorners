import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings

warnings.simplefilter("ignore")


def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs, use_grid_search=True,
                     best_components=None, best_params=None):
    start = time()

    if use_grid_search == True:

        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)

        grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
        grid_obj.fit(X_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:

        estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)
        best_pipe = pipeline.fit(X_train, y_train)

    end = time()

    print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    return best_pipe


def predict_labels(clf, best_pipe, features, target):
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    end = time()

    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target.values, y_pred)


def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                            params, scorer, jobs,
                            use_grid_search=True, **kwargs):
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))

    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)

    print("Calibrating probabilities of classifier...")
    start = time()
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv='prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         predict_labels(clf, best_pipe, X_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__,
                                                     predict_labels(clf, best_pipe, X_test, y_test)))

    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(
        clf, best_pipe, X_test, y_test)


def plot_confusion_matrix(y_test, X_test, clf, dim_reduce, cmap=plt.cm.Blues, normalize=False):
    labels = [">=5", "<5"]
    cm = confusion_matrix(y_test, clf.predict(dim_reduce.transform(X_test)), labels)

    if normalize == True:
        cm = cm.astype('float') / cm.sum()

    sns.set_style("whitegrid", {"axes.grid": False})
    plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = "Confusion matrix of a {} with {}".format(best_clf.base_estimator.__class__.__name__,
                                                      best_dm_reduce.__class__.__name__)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

    y_pred = clf.predict(dim_reduce.transform(X_test))
    print(classification_report(y_test, y_pred))


def explore_data(features, inputs):
    i = 1
    for col in features.columns:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 1})

        plt.subplot(3, 2, 0 + i)
        j = i - 1

        sns.distplot(inputs[inputs['label'] == '>=5'].iloc[:, j], hist=False, label='>=5')
        sns.distplot(inputs[inputs['label'] == '<5'].iloc[:, j], hist=False, label='<5')
        plt.legend();
        i = i + 1

    plt.show()

    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    feature_details = features.describe().transpose()

    return feature_details


def find_best_classifier(clasiffiers, dm_reductions, scorer, X_t, y_t, X_c, y_c, X_v, y_v, cv_sets, params, jobs):
    clfs_return = []
    dm_reduce_return = []
    train_scores = []
    test_scores = []

    for dm in dm_reductions:

        for clf in clfs:
            clf, dm_reduce, train_score, test_score = train_calibrate_predict(clf=clf, dm_reduction=dm, X_train=X_t,
                                                                              y_train=y_t,
                                                                              X_calibrate=X_c, y_calibrate=y_c,
                                                                              X_test=X_v, y_test=y_v, cv_sets=cv_sets,
                                                                              params=params[clf], scorer=scorer,
                                                                              jobs=jobs, use_grid_search=True)

            clfs_return.append(clf)
            dm_reduce_return.append(dm_reduce)
            train_scores.append(train_score)
            test_scores.append(test_score)

    return clfs_return, dm_reduce_return, train_scores, test_scores


def plot_training_results(clfs, dm_reductions, train_scores, test_scores):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})
    ax = plt.subplot(111)
    w = 0.5
    x = np.arange(len(train_scores))
    ax.set_yticks(x + w)
    ax.legend((train_scores[0], test_scores[0]), ("Train Scores", "Test Scores"))
    names = []

    for i in range(0, len(clfs)):
        clf = clfs[i]
        clf_name = clf.base_estimator.__class__.__name__
        dm = dm_reductions[i]
        dm_name = dm.__class__.__name__

        name = "{} with {}".format(clf_name, dm_name)
        names.append(name)

    ax.set_yticklabels((names))
    plt.xlim(0.55, 0.65)
    plt.barh(x, test_scores, color='b', alpha=0.6)
    plt.title("Test Data Accuracy Scores")
    plt.figure(1)

    plt.show()


n_jobs = 1

inputs = read_csv("data/corners3.txt")
rows_to_drop = ['Match date', 'Tournament', 'Home team', 'Home team corners', 'Away team corners', 'Away team',
                'Home team corners kicked', 'Away team corners concieved']
inputs.dropna(inplace=True)
inputs = inputs.drop(rows_to_drop, axis=1)

labels = inputs.loc[:, 'label']
features = inputs.drop('label', axis=1)

feature_details = explore_data(features, inputs)

X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size=0.2,
                                                                        random_state=42,
                                                                        stratify=labels)
X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size=0.3,
                                                              random_state=42,
                                                              stratify=y_train_calibrate)

cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
cv_sets.get_n_splits(X_train, y_train)

RF_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
AB_clf = AdaBoostClassifier(n_estimators=200, random_state=2)
GNB_clf = GaussianNB()
KNN_clf = KNeighborsClassifier()
LOG_clf = linear_model.LogisticRegression(multi_class="ovr", solver="sag", class_weight='balanced')
clfs = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf]

feature_len = features.shape[1]
scorer = make_scorer(accuracy_score)
parameters_RF = {'clf__max_features': ['auto', 'log2'],
                 'dm_reduce__n_components': np.arange(5, feature_len, np.around(feature_len))}
parameters_AB = {'clf__learning_rate': np.linspace(0.5, 2, 5),
                 'dm_reduce__n_components': np.arange(5, feature_len, np.around(feature_len))}
parameters_GNB = {'dm_reduce__n_components': np.arange(5, feature_len, np.around(feature_len))}
parameters_KNN = {'clf__n_neighbors': [3, 5, 10],
                  'dm_reduce__n_components': np.arange(5, feature_len, np.around(feature_len))}
parameters_LOG = {'clf__C': np.logspace(1, 1000, 5),
                  'dm_reduce__n_components': np.arange(5, feature_len, np.around(feature_len))}

parameters = {clfs[0]: parameters_RF,
              clfs[1]: parameters_AB,
              clfs[2]: parameters_GNB,
              clfs[3]: parameters_KNN,
              clfs[4]: parameters_LOG}

pca = PCA()
dm_reductions = [pca]

clf = LOG_clf
clf.fit(X_train, y_train)
print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                     accuracy_score(y_train, clf.predict(X_train))))
print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test))))

clfs, dm_reductions, train_scores, test_scores = find_best_classifier(clfs, dm_reductions, scorer, X_train, y_train,
                                                                      X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                                                                      parameters, n_jobs)

plot_training_results(clfs, dm_reductions, np.array(train_scores), np.array(test_scores))

best_clf = clfs[np.argmax(test_scores)]
best_dm_reduce = dm_reductions[np.argmax(test_scores)]
print("The best classifier is a {} with {}.".format(best_clf.base_estimator.__class__.__name__,
                                                    best_dm_reduce.__class__.__name__))
plot_confusion_matrix(y_test, X_test, best_clf, best_dm_reduce, normalize=True)

## Enter your own data

# data = [[6, 5, 5, 5, 15, 3]]
# game = pd.DataFrame(data, columns=['Home team corners average', 'Corners >5 home team','Corners >5 concieved away team', 'Away team corners concieved average', 'Corners difference', 'Corners >5 against home team'])
# print(best_clf.predict(best_dm_reduce.transform(game)))
# print(best_clf.predict_proba(best_dm_reduce.transform(game)))
