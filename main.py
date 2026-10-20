import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cnf 
import utils as utl
import DeepLearning as DL


# Перебор гиперпараметров
def build_combo_param():
    combo_param = utl.parametrs(cnf.RandomForestParam)
    combo_param += utl.parametrs(cnf.LinRegParam)
    combo_param += utl.parametrs(cnf.KNNParam)
    combo_param += utl.parametrs(cnf.DecisionTreeParam)
    combo_param += utl.parametrs(cnf.LGBMParam)
    combo_param += utl.parametrs(cnf.XGBoostParam)
    combo_param += utl.parametrs(cnf.CatBoostParam)
    combo_param += utl.parametrs(cnf.DLParam)
    return combo_param

# ------------------ Цикл валидации ------------------
def run_validation(X, X_ohe, y, combo_param):
    results = []
    dl_curves = [] 
    ensemble_result = pd.DataFrame(index=y.index)
    K_Fold = KFold(n_splits=cnf.n_splits, shuffle=True, random_state=cnf.random_state)
    
    # Перебор всех гиперпараметров
    for k, combo in enumerate(combo_param):
        
        ensemble_result_pred = np.zeros(len(y))
        
        params = combo.copy()
        name_model = params.pop('name_model')
        use_scaler = params.pop('use_scaler')
        use_category = params.pop('use_category')

        # Флаг на категориальные признаки - если False, делаем OneHotCoding
        X_cur = X if use_category else X_ohe

        array_train_RMSE = []
        array_val_RMSE = []

        # Запускаем DL
        if name_model == 'nn_model':
            
            for i, (train_index, val_index) in enumerate(K_Fold.split(X, y)):

                # Объявляем модель
                loss_model = nn.MSELoss()
                torch.manual_seed(cnf.random_state)

                model = DL.nn_model(X_cur.shape[1])
                opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                opt
                                                                , mode=params['mode']
                                                                , factor=params['factor']
                                                                , patience=params['patience']
                                                                , threshold=params['threshold']
                                                                , min_lr=params['min_lr']
                                                                )

                # Подготовка данных для обучения
                X_train, y_train, X_val, y_val = utl.preparation_train(X_cur, y, train_index, val_index, use_scaler)

                nn.init.constant_(model[-1].bias, float(y_train.mean()))
                
                tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
                tensor_X_val  = torch.tensor(X_val,  dtype=torch.float32)
                tensor_y_train = torch.tensor(y_train.values, dtype=torch.float32)
                tensor_y_val  = torch.tensor(y_val.values,  dtype=torch.float32)

                dataset = TensorDataset(tensor_X_train, tensor_y_train)
                dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

                # Цикл по эпохам
                train_loop = tqdm(range(params['EPOCHS']))
                for epoch in train_loop:

                    model.train()

                    for _, val in enumerate(dataloader):
                    
                        # Прямой проход
                        pred = model(val[0]).squeeze(1)
                        loss = loss_model(pred, val[1])

                        # Обратный проход
                        opt.zero_grad()
                        loss.backward()

                        # Шаг оптимизации
                        opt.step()

                    model.eval()

                    # Считаем на валидации
                    with torch.no_grad():
                        # Считаем предсказания
                        pred_train = model(tensor_X_train).squeeze(1)
                        pred_val = model(tensor_X_val).squeeze(1)

                        # Считаем лосс
                        loss_train = loss_model(pred_train, tensor_y_train)
                        loss_val = loss_model(pred_val, tensor_y_val)

                        # Считаем точность на трейне
                        train_loss = loss_train.item()
                        train_RMSE = train_loss ** 0.5

                        # Считаем точность на тесте
                        val_loss = loss_val.item()
                        val_RMSE = val_loss ** 0.5

                    train_loop.set_description(f"Epoch [{epoch+1}/{params['EPOCHS']}], train_loss={train_loss:2f}, train_RMSE={train_RMSE:2f}, val_loss={val_loss:2f} val_RMSE={val_RMSE:2f} lr={scheduler.get_last_lr()[0]}")

                    scheduler.step(train_loss)


                    dl_curves.append({

                        'name_model': name_model
                        , 'Fold': i
                        , 'epoch': epoch
                        , 'train_RMSE': train_RMSE  
                        , 'train_loss': train_loss
                        , 'val_RMSE': val_RMSE
                        , 'val_loss': val_loss
                        , 'lr': opt.param_groups[0]['lr']
                        , 'combo': f"weight_decay={params['weight_decay']}, factor={params['factor']}"
                    })

                ensemble_result_pred[val_index] = pred_val.numpy()
                
                array_train_RMSE.append(train_RMSE)   # значение последней эпохи
                array_val_RMSE.append(val_RMSE)


# Запускаем линейные модели и деревья
        else:     
            if name_model == 'CatBoost':
                params['cat_features'] = X_cur.select_dtypes('category').columns.tolist()
            
            train_loop = tqdm(enumerate(K_Fold.split(X, y)))
            for i, (train_index, val_index) in train_loop:
                
                # Подготовка данных для обучения
                X_train, y_train, X_val, y_val = utl.preparation_train(X_cur, y, train_index, val_index, use_scaler)

                model = cnf.models[name_model](**params)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                ensemble_result_pred[val_index] = y_val_pred
                
                # Считаем RMSE
                train_RMSE = root_mean_squared_error(y_train, y_train_pred)
                val_RMSE = root_mean_squared_error(y_val, y_val_pred)

                array_train_RMSE.append(train_RMSE)
                array_val_RMSE.append(val_RMSE)
                                
                train_loop.set_description(f"K_Fold [{i+1}/{cnf.n_splits}], model {name_model}, train_RMSE={np.mean(array_train_RMSE):2f}, val_RMSE={np.mean(array_val_RMSE):2f}")
        
        # Считаем RMSE для текущих модель-гиперпараметры
        mean_train_RMSE = float(np.mean(array_train_RMSE))
        mean_train_std = float(np.std(array_train_RMSE))
        mean_val_RMSE = float(np.mean(array_val_RMSE))
        mean_val_std = float(np.std(array_val_RMSE))
        
        ensemble_result[f'{name_model}_{k}'] = ensemble_result_pred
        # Переименовываем параметры в Катбусте
        combo_row = {cnf.rename_params.get(k, k): v for k, v in combo.items()}

        results.append({
            **combo_row
            , 'col': f'{name_model}_{k}' 
            , 'train_RMSE'  :   mean_train_RMSE
            , 'train_std'       :   mean_train_std
            , 'val_RMSE'   :   mean_val_RMSE
            , 'val_std'        :   mean_val_std
        })
    
    # Выгрузка результатов и обучения ДЛ в эксель
    return utl.save_validation(results, dl_curves, cnf.path['file_path'], cnf.exp_name), ensemble_result.to_csv(cnf.path['file_path'] / f'ensemble_result_{cnf.exp_name}.csv', index=False)
    
    




# ------------------ Цикл инференеса ------------------
def run_inference(X, X_ohe, y, X_test, X_ohe_test, df_test, combo_param):
    
    submissions = pd.DataFrame({'Id': df_test['Id']})
    results_test = []

    # Перебор гиперпараметров
    for k, combo in enumerate(combo_param):
        params = combo.copy()
        name_model = params.pop('name_model')
        use_scaler = params.pop('use_scaler')
        use_category = params.pop('use_category')
        
        # Обучение ДЛ
        if name_model == 'nn_model':
            
            # Подготовка данных
            X_fit, X_pred = utl.preparation_inference(
                                X if use_category else X_ohe,
                                X_test if use_category else X_ohe_test,
                                use_scaler)
            
            tensor_X_fit  = torch.tensor(np.asarray(X_fit,  dtype=np.float32))
            tensor_X_pred = torch.tensor(np.asarray(X_pred, dtype=np.float32))
            tensor_y      = torch.tensor(y.values, dtype=torch.float32)
            
            
            # Объявляем модель
            loss_model = nn.MSELoss()
            torch.manual_seed(cnf.random_state)

            model = DL.nn_model(X_fit.shape[1])
            nn.init.constant_(model[-1].bias, float(y.mean()))
            opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            opt
                                                            , mode=params['mode']
                                                            , factor=params['factor']
                                                            , patience=params['patience']
                                                            , threshold=params['threshold']
                                                            , min_lr=params['min_lr']
                                                            )


            dataset = TensorDataset(tensor_X_fit, tensor_y)
            dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

            train_loop = tqdm(range(params['EPOCHS']))
            for epoch in train_loop:

                model.train()

                for _, val in enumerate(dataloader):
                    
                    # Прямой проход
                    pred = model(val[0]).squeeze(1)
                    loss = loss_model(pred, val[1])

                    # Обратный проход
                    opt.zero_grad()
                    loss.backward()

                    # Шаг оптимизации
                    opt.step()

                model.eval()

                # Считаем на валидации
                with torch.no_grad():
                    # Считаем предсказания
                    pred_train = model(tensor_X_fit).squeeze(1)

                    # Считаем лосс
                    loss_train = loss_model(pred_train, tensor_y)

                    # Считаем точность на трейне
                    train_loss = loss_train.item()
                    train_RMSE = train_loss ** 0.5

                train_loop.set_description(f"Epoch [{epoch+1}/{params['EPOCHS']}], train_loss={train_loss:2f}, train_RMSE={train_RMSE:2f}, lr={scheduler.get_last_lr()[0]}")

                scheduler.step(train_loss)


            model.eval()
            # Считаем итоговый train_RMSE и получаем ответы модели на тесте
            with torch.no_grad():
                train_RMSE = loss_model(model(tensor_X_fit).squeeze(1), tensor_y).item() ** 0.5
                answer = np.expm1(model(tensor_X_pred).squeeze(1).numpy())
            
            
            col = f'{name_model}_{k}'
            submissions[col] = answer
            combo_row = {cnf.rename_params.get(kk, kk): v for kk, v in combo.items()}
            results_test.append({
                    **combo_row
                    , 'col': col
                    , 'train_RMSE': train_RMSE
                })           
            
        # Линейные модели и деревья
        else:
            
            # Флаг на категориальные признаки - если False, делаем OneHotCoding
            # Подготовка данных
            X_fit, X_pred = utl.preparation_inference(
                                X if use_category else X_ohe,
                                X_test if use_category else X_ohe_test,
                                use_scaler)
            if name_model == 'CatBoost':
                params['cat_features'] = X_fit.select_dtypes('category').columns.tolist()
                
            model = cnf.models[name_model](**params).fit(X_fit, y)
            
            y_train_pred = model.predict(X_fit)

            # Считаем RMSE
            train_RMSE = root_mean_squared_error(y, y_train_pred)

            # Сохраняем данные
            col = f'{name_model}_{k}'
            submissions[col] = np.expm1(model.predict(X_pred))
            # Переименовываем Катбуст
            combo_row = {cnf.rename_params.get(kk, kk): v for kk, v in combo.items()}

            results_test.append({
                    **combo_row
                    , 'col': col
                    , 'train_RMSE'  :   train_RMSE
                })
            
            print(f"model {name_model}, iter {k+1}/{len(combo_param)}, train_RMSE={train_RMSE:2f}")
                    
            
    # Выгружаем результат и ответы моделей
    return utl.save_inference(submissions, results_test, cnf.path['file_path'], cnf.exp_name)


def main():
    assert cnf.mode in ('validation', 'inference')

    df_train = pd.read_csv(cnf.path['train_path'])
    df_test  = pd.read_csv(cnf.path['test_path'])
    X, X_ohe, y, X_test, X_ohe_test = utl.dataset_forming(df_train, df_test)
    combo_param = build_combo_param()

    if cnf.mode == 'validation':
        out = run_validation(X, X_ohe, y, combo_param)
    else:
        out = run_inference(X, X_ohe, y, X_test, X_ohe_test, df_test, combo_param)

    print(out)


if __name__ == '__main__':
    main()
