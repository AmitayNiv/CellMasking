

def train_classifier(args,device,wandb_exp):
    
    X_train, y_train, x_validate_, y_validate_, X_test, y_test = data_obj.x_train, data_obj.y_train, data_obj.x_validate_, data_obj.y_validate_, data_obj.x_test, data_obj.y_test
    # print_statistics = True
    # print("Start Training To Estimate: [{}]".format(model_obj["name"]))
    # print("Total number of gens for this train: [{}], Total number of features: [{}]".format((model_obj["Nor_y"]).shape[0],model_obj["x"].shape[1]))

    print("Total number of features: [{}]".format(X_train.shape[1]))
    print("Train Size: [{}]".format(y_train.shape[0]))
    print("Validation Size: [{}]".format(y_validate_.shape[0]))
    print("Test Size: [{}]".format(y_test.shape[0]))

    train_losses = []
    val_losses =[]
    best_model_auc = 0
    # check if the tables are at the same length
    assert(X_train.shape[0]==len(y_train))
    assert(X_test.shape[0]==len(y_test))
    assert(x_validate_.shape[0]==len(y_validate_))

    train_batch = np.array_split(X_train, 50)
    label_batch = np.array_split(y_train, 50)

    for i in range(len(train_batch)):
        train_batch[i] = torch.from_numpy(train_batch[i]).float()
    for i in range(len(label_batch)):
        label_batch[i] = torch.from_numpy(label_batch[i]).float().view(-1, 1)


    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().view(-1, 1).to(device)
    x_validate_ = torch.from_numpy(x_validate_).float().to(device)
    y_validate_ = torch.from_numpy(y_validate_).float().view(-1, 1).to(device)
    if model == None:
        model = Model(X_train.shape[1]) #binaryClassification(X_train.shape[1])#SGDClassifier({'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight},random_state=RANDOM_SEED)    #Model(X_train.shape[1])
        model = model.to(device)
    criterion = nn.BCELoss()# nn.SmoothL1Loss()# nn.SmoothL1Loss()#  #nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=float(args.lr),weight_decay=float(args.weight_decay),momentum=0.9) # optim.Adam(model.parameters(), lr=args.lr) #lr=0.00001)#optim.SGD(model.parameters(),lr=float(args.lr),weight_decay=float(args.weight_decay),momentum=0.9) #
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # print("Train set size:[{}], Test set size:[{}]".format(X_train.shape[0],X_test.shape[0])) 
    for e in range(args.epochs):
        model.train()
        train_loss = 0
        for i in range(len(train_batch)):
            optimizer.zero_grad()
            x = train_batch[i].to(device)
            y = label_batch[i].to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        else:
            val_loss = 0
        
            with torch.no_grad():
                model.eval()
                predictions = model(x_validate_)
                val_loss +=  criterion(predictions, y_validate_).item()
                predictions = predictions.detach().cpu().numpy()
                y_pred_val = (predictions> 0.5).astype(int)
                val_score = evaluate(y_validate_, y_pred_val, predictions)

    
                    
            train_losses.append(train_loss/len(train_batch))
            val_losses.append(val_loss)

            print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                  "Training Loss: {:.5f}.. ".format(train_loss/len(train_batch)),
                  "Val Loss: {:.5f}.. ".format(val_loss),
                  "Val AUC: {:.5f}.. ".format(val_score["auc"]),
                  "count {}/{}".format(sum(y_pred_val),len(y_pred_val)))
            if val_score["auc"] >best_model_auc:
                best_model_auc = val_score["auc"]
                best_model_index = e+1
                # best_model_name = "/F_model_{:.3f}.pt".format(best_model_auc)
                # best_model_path = current_folder+best_model_name
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                # torch.save(model,best_model_path)
                best_model = copy.deepcopy(model)
            # scheduler.step(val_score["auc"])

    best_model = best_model.cuda()#torch.load(best_model_path).cuda()
    with torch.no_grad():
      best_model.eval()
      y_pred_score = best_model(X_test)
      y_pred_score = y_pred_score.detach().cpu().numpy()
      y_pred_test = (y_pred_score> 0.5).astype(int)
      test_score = evaluate(y_test, y_pred_test, y_pred_score)
      print("Classifier test results:")
      print(test_score)
      return best_model

