import pickle
import sklearn
import xgboost as xgb

print('importing model...')

#model_name = 

with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

print('...')
print('Finished importing model...')
print('...')

profile = {
    'parents': 'great_pret',
    'has_nurs': 'improper',
    'form': 'completed',
    'children': '2',
    'housing': 'less_conv',
    'finance': 'inconv',
    'social': 'problematic',
    'health': 'priority'
}

features = list(dv.get_feature_names_out())
X = dv.transform(profile)
dX = xgb.DMatrix(X, feature_names=features)
class_of_application = model.predict(dX)[0]

n = class_of_application

if n == 0:
    print('This application is not_recommeded')
elif n == 1:
    print('This application is priority')
elif n == 2:
    print('This application is recommended')
elif n == 3:
    print('This application is Special_priority')
else:
    print('This application is very_recommeded')


