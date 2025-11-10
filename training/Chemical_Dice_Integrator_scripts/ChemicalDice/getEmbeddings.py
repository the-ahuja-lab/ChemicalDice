from ChemicalDice.architecture import *                              # Contains the model architecture
from ChemicalDice.myImports import *                                 # Contains all import files

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest

from sklearn.impute import SimpleImputer

seed_value = 42
batch_size = 64

def init():
    rs = RandomState(MT19937(SeedSequence(seed_value))) 
    np.random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)



class MyDataset(Dataset):
  def __init__(self, X1,X2, X3, X4, X5,  X6, y):
    self.X1 = X1
    self.X2 = X2
    self.X3 = X3
    self.X4 = X4
    self.X5 = X5
    self.X6 = X6
    
    self.y = y
    
    self.n_samples = X1.shape[0]

  def __getitem__(self, index):
    k1 = self.X1[index]
    k2 = self.X2[index]
    k3 = self.X3[index]
    k4 = self.X4[index]
    k5 = self.X5[index]
    k6 = self.X6[index]

    label = self.y[index]
    
    return torch.tensor(k1, dtype=torch.float32), torch.tensor(k2, dtype=torch.float32), torch.tensor(k3, dtype=torch.float32), torch.tensor(k4, dtype=torch.float32), torch.tensor(k5, dtype=torch.float32), torch.tensor(k6, dtype=torch.float32), torch.tensor(label) 

  def __len__(self):
    return self.n_samples
  

from sklearn.preprocessing import StandardScaler



def AutoencoderReconstructor_training_8192(X1, X2, X3 ,X4 ,X5 , X6,CDI_epochs,CDI_k):
    
    scaler =StandardScaler()
    
    ids = X1.index.to_numpy()

    embd_sizes =[X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1],X5.shape[1],X6.shape[1]]

    def sum_except_self(nums):
        total_sum = sum(nums)
        return [total_sum - num for num in nums]

    embd_sizes_sum = sum_except_self(embd_sizes)

    # print(embd_sizes_sum)
    # print(embd_sizes)

    
    # X1 = scaler.fit_transform(X1)
    # X2 = scaler.fit_transform(X2)
    # X3 = scaler.fit_transform(X3)
    # X4 = scaler.fit_transform(X4)
    # X5 = scaler.fit_transform(X5)
    # X6 = scaler.fit_transform(X6)

    X1 = X1.values
    X2 = X2.values
    X3 = X3.values
    X4 = X4.values
    X5 = X5.values
    X6 = X6.values

    
    _, X1_dim = X1.shape
    _, X2_dim = X2.shape
    _, X3_dim = X3.shape
    _, X4_dim = X4.shape
    _, X5_dim = X5.shape
    _, X6_dim = X6.shape

    # print(X1.shape)
    # print(X2.shape)
    # print(X3.shape)
    # print(X4.shape)
    # print(X5.shape)
    # print(X6.shape)

    

    # print(X1.shape[0])


    data = MyDataset(np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6), np.array([-1] * X1.shape[0]))
    data_loader = DataLoader(data, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn) 
    
    train_loss_values = []
    val_loss_values = []
    loss_lst = []



    embed_dim = 8192
    latent_space_dims = [X1_dim, X2_dim, X3_dim, X4_dim, X5_dim, X6_dim]
    # print('k=', embed_dim)

    net_cdi = ChemicalDiceIntegrator(latent_space_dims=latent_space_dims, embedding_dim=embed_dim, embd_sizes=embd_sizes, embd_sizes_sum=embd_sizes_sum,k=CDI_k).to(device)

    net_cdi, loss_train, loss_val, _, _ ,embeddings_df= trainAE_8192(net_cdi, "AER_8192", embed_dim, data_loader, data_loader, CDI_epochs, True)

    #net_cdi.load_state_dict(torch.load(f"../weights/trainAE_{dataset_name}/{embed_dim}_cdi.pt"))
    #net_cdi.load_state_dict(torch.load(f"AER_8192_cdi.pt"))
    

    # def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
    #   if not os.path.exists(file_path):
    #       os.makedirs(file_path) 
    #   with open(file_path + file_name, mode='w', newline='') as file:
    #       writer = csv.writer(file) 
    #       for row in range(ids.shape[0]):
    #           writer.writerow(np.concatenate(([ids[row]], embeddings[row])))
    #print(ids.shape, embeddings.shape)
    #writeEmbeddingsIntoFile('../embeddings/', f'{dataset_name}_embed_{embed_dim}.csv', ids, embeddings)
    # model = torch.load(f"../weights/trainAE_{dataset_name}/{embed_dim}_cdi.pt")
    # encoder_1_weights = model['encoders.6.encoder.0.weight']
    # row_means = torch.mean(encoder_1_weights, dim=0)
    # Convert the tensor into a NumPy array
    #row_means_np = row_means.cpu().numpy()
    #print(ids.shape)
    embeddings_df = pd.DataFrame(embeddings_df)
    embeddings_df['id'] = ids
    embeddings_df
    return embeddings_df#, model_wt


def AutoencoderReconstructor_training_other(X1, X2, X3 ,X4 ,X5 , X6, embedding_sizes,CDI_epochs,CDI_k):
    scaler =StandardScaler()
    
    ids = X1.index.to_numpy()

    embd_sizes =[X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1],X5.shape[1],X6.shape[1]]

    def sum_except_self(nums):
        total_sum = sum(nums)
        return [total_sum - num for num in nums]

    embd_sizes_sum = sum_except_self(embd_sizes)

    # print(embd_sizes_sum)
    # print(embd_sizes)

    # X1 = scaler.fit_transform(X1)
    # X2 = scaler.fit_transform(X2)
    # X3 = scaler.fit_transform(X3)
    # X4 = scaler.fit_transform(X4)
    # X5 = scaler.fit_transform(X5)
    # X6 = scaler.fit_transform(X6)


    X1 = X1.values
    X2 = X2.values
    X3 = X3.values
    X4 = X4.values
    X5 = X5.values
    X6 = X6.values
    
    _, X1_dim = X1.shape
    _, X2_dim = X2.shape
    _, X3_dim = X3.shape
    _, X4_dim = X4.shape
    _, X5_dim = X5.shape
    _, X6_dim = X6.shape

    # print(X1.shape)
    # print(X2.shape)
    # print(X3.shape)
    # print(X4.shape)
    # print(X5.shape)
    # print(X6.shape)

    # print(X1.shape[0])


    data = MyDataset(np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6), np.array([-1] * X1.shape[0]))
    data_loader = DataLoader(data, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn) 

    train_loss_values = []
    val_loss_values = []
    loss_lst = []
    all_embeddings = []

    embed_dim = embedding_sizes
    latent_space_dims = [X1_dim, X2_dim, X3_dim, X4_dim, X5_dim, X6_dim]
    #print('k=', embed_dim)
    net_cdi = ChemicalDiceIntegrator(latent_space_dims=latent_space_dims, embedding_dim=8192, embd_sizes=embd_sizes, embd_sizes_sum=embd_sizes_sum,k=CDI_k).to(device)
    #net_cdi, loss_train, loss_val, _, _,model_state = trainAE(net_cdi, "test", embed_dim, data_loader, data_loader, 5, True)
    net_cdi.load_state_dict(torch.load("AER_8192_cdi.pt"))

    #net_cdi.load_state_dict(model_state)
    # finetune_cdi, loss_train, loss_val, _, _ = finetune_AE(finetune_cdi, dataset_name, user_embed_size, choice, data_loader, data_loader, 2, True)
    finetune_cdi = FineTuneChemicalDiceIntegrator(net_cdi, user_embed_dim=embed_dim, default_embed_dim=8192).to(device)
    finetune_cdi, loss_train, loss_val, _, _ = finetune_AE(finetune_cdi, user_embed_dim=embed_dim, choice=1, train_loader=data_loader, epochs=CDI_epochs, verbose=True)


    embeddings = None
    finetune_cdi.eval()
    for data in data_loader:
        k1, k2, k3, k4, k5, k6, _ = data
        k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
        embed = finetune_cdi.getEmbed([k1, k2, k3, k4, k5, k6])
        # print(output.shape)
        
        if embeddings is None:
            embeddings = embed.cpu().detach().numpy()
        else:
            embeddings = np.append(embeddings, embed.cpu().detach().numpy(), axis = 0)

        # def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
        #   if not os.path.exists(file_path):
        #       os.makedirs(file_path) 
        #   with open(file_path + file_name, mode='w', newline='') as file:
        #       writer = csv.writer(file) 
        #       for row in range(ids.shape[0]):
        #           writer.writerow(np.concatenate(([ids[row]], embeddings[row])))
        #print(ids.shape, embeddings.shape)
        #writeEmbeddingsIntoFile('../embeddings/', f'{dataset_name}_embed_{embed_dim}.csv', ids, embeddings)
        # model = torch.load(f"../weights/trainAE_{dataset_name}/{embed_dim}_cdi.pt")
        # encoder_1_weights = model['encoders.6.encoder.0.weight']
        # row_means = torch.mean(encoder_1_weights, dim=0)
        # Convert the tensor into a NumPy array
        #row_means_np = row_means.cpu().numpy()
        #print(ids.shape)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['id'] = ids
    return embeddings_df



def AutoencoderReconstructor_training_single(X1, X2, X3 ,X4 ,X5 , X6,embed_dim,CDI_epochs,CDI_k):
    
    scaler =StandardScaler()
    
    ids = X1.index.to_numpy()

    embd_sizes =[X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1],X5.shape[1],X6.shape[1]]

    def sum_except_self(nums):
        total_sum = sum(nums)
        return [total_sum - num for num in nums]

    embd_sizes_sum = sum_except_self(embd_sizes)

    # print(embd_sizes_sum)
    # print(embd_sizes)

    
    # X1 = scaler.fit_transform(X1)
    # X2 = scaler.fit_transform(X2)
    # X3 = scaler.fit_transform(X3)
    # X4 = scaler.fit_transform(X4)
    # X5 = scaler.fit_transform(X5)
    # X6 = scaler.fit_transform(X6)

    X1 = X1.values
    X2 = X2.values
    X3 = X3.values
    X4 = X4.values
    X5 = X5.values
    X6 = X6.values

    
    _, X1_dim = X1.shape
    _, X2_dim = X2.shape
    _, X3_dim = X3.shape
    _, X4_dim = X4.shape
    _, X5_dim = X5.shape
    _, X6_dim = X6.shape

    # print(X1.shape)
    # print(X2.shape)
    # print(X3.shape)
    # print(X4.shape)
    # print(X5.shape)
    # print(X6.shape)

    

    # print(X1.shape[0])


    data = MyDataset(np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6), np.array([-1] * X1.shape[0]))
    data_loader = DataLoader(data, batch_size = batch_size, shuffle=False, worker_init_fn=worker_init_fn) 
    
    train_loss_values = []
    val_loss_values = []
    loss_lst = []



    latent_space_dims = [X1_dim, X2_dim, X3_dim, X4_dim, X5_dim, X6_dim]
    # print('k=', embed_dim)

    net_cdi = ChemicalDiceIntegrator(latent_space_dims=latent_space_dims, embedding_dim=embed_dim, embd_sizes=embd_sizes, embd_sizes_sum=embd_sizes_sum,k=CDI_k).to(device)

    net_cdi, loss_train, loss_val, _, _ , model_wt= trainAE_8192(net_cdi, "AER_"+str(embed_dim), embed_dim, data_loader, data_loader, CDI_epochs, True)
    # print(0)
    #net_cdi.load_state_dict(torch.load(f"../weights/trainAE_{dataset_name}/{embed_dim}_cdi.pt"))
    net_cdi.load_state_dict(torch.load("AER_"+str(embed_dim)+"_cdi.pt"))
    # print(1)

    embeddings = None
    net_cdi.eval()
    for data in data_loader:
        k1, k2, k3, k4, k5, k6, _ = data
        k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)

        _, _, _, _, _, _, _, _, _, _, _, _, output, _ = net_cdi.forward([k1, k2, k3, k4, k5, k6])
        # print(output.shape)

        if embeddings is None:
            embeddings = output.cpu().detach().numpy()

        else:
            embeddings = np.append(embeddings, output.cpu().detach().numpy(), axis = 0)


    # def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
    #   if not os.path.exists(file_path):
    #       os.makedirs(file_path) 
    #   with open(file_path + file_name, mode='w', newline='') as file:
    #       writer = csv.writer(file) 
    #       for row in range(ids.shape[0]):
    #           writer.writerow(np.concatenate(([ids[row]], embeddings[row])))
    #print(ids.shape, embeddings.shape)
    #writeEmbeddingsIntoFile('../embeddings/', f'{dataset_name}_embed_{embed_dim}.csv', ids, embeddings)
    # model = torch.load(f"../weights/trainAE_{dataset_name}/{embed_dim}_cdi.pt")
    # encoder_1_weights = model['encoders.6.encoder.0.weight']
    # row_means = torch.mean(encoder_1_weights, dim=0)
    # Convert the tensor into a NumPy array
    #row_means_np = row_means.cpu().numpy()
    #print(ids.shape)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['id'] = ids
    # print(embeddings_df)
    return embeddings_df, model_wt








seed_value = 42
rs = RandomState(MT19937(SeedSequence(seed_value))) 
np.random.seed(seed_value)
batch_size = 64

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def count_items(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_values, counts))

def convert_to_list(lst):
    ans = []
    for i in range(len(lst)):
        ans.append(lst[i].item())
    return ans

#trainAE(net_cdi, "test", embed_dim, data_loader, data_loader, 2, True)
def trainAE_8192(model, dataset_name, embed_dim, train_loader, val_loader, epochs, verbose = False):
    
    # These are the parameters
    
    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.MSELoss()
    LEARNING_RATE = 5e-1
    WEIGHT_DECAY = 0
    OPTIMIZER = optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience = 5, verbose = True)

    loss_train = []
    loss_val = []

    acc_train = []
    acc_val = []

    alpha, beta, gamma = 0.33, 0.33, 0.33
    # alpha, beta = 0.5, 0.5

    for epoch in range(NUM_EPOCHS):

        st = time.time()
        model.train()
        # print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0 
        average_batch_loss_train = 0

        for i, data in enumerate(train_loader, 0):
            OPTIMIZER.zero_grad()
            
            k1, k2, k3, k4, k5, k6, labels = data

            k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            labels = labels.to(device)

            key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
            key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
            key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
            key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
            key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
            key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)

            key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])

            encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
            encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
            encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
            encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
            encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
            encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

            reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
            reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
            reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
            reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
            reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
            reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

            concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
            reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

            total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
            total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6
            # print('train',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
            total_loss_encoder = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)

            # print(device)
            total_loss_encoder.backward()
            OPTIMIZER.step()

            _loss = total_loss_encoder.item()

            batch_loss_train += _loss

            average_batch_loss_train = batch_loss_train / (i+1)

        loss_train.append(average_batch_loss_train)

        SCHEDULER.step(average_batch_loss_train)


        ## Validation
        batch_loss_val = 0
        avg_loss_val = 0

        # with torch.no_grad():
        #     model.eval()
        #     for i, data in enumerate(val_loader):
        #         k1, k2, k3, k4, k5, k6, labels = data
        #         k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
        #         labels = labels.to(device)

        #         key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
        #         key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
        #         key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
        #         key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
        #         key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
        #         key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)
                
        #         key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])
            
        #         encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
        #         encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
        #         encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
        #         encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
        #         encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
        #         encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

        #         reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
        #         reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
        #         reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
        #         reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
        #         reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
        #         reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

        #         concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
        #         reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

        #         total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
        #         total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6

        #         # print('val',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
        #         total_loss_encoder_val = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)
                
        #         _loss_val = total_loss_encoder_val.item()
        #         batch_loss_val += _loss_val
        #         avg_loss_val = batch_loss_val / (i+1)

        #     loss_val.append(avg_loss_val)
        if verbose:
            print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} time: {time.time() - st}')
            # print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')
    embeddings = None
    model.eval()
    for data in train_loader:
        k1, k2, k3, k4, k5, k6, _ = data
        k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)

        _, _, _, _, _, _, _, _, _, _, _, _, output, _ = model.forward([k1, k2, k3, k4, k5, k6])
        # print(output.shape)

        if embeddings is None:
            embeddings = output.cpu().detach().numpy()

        else:
            embeddings = np.append(embeddings, output.cpu().detach().numpy(), axis = 0)

    # print('Training FINISHED')
    print(f'Training AUTOENCODER finished Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')


    torch.save(model.state_dict(), f"{dataset_name}_cdi.pt")

    # model_state = model.state_dict()
    # mopac_feature_wt = model_state['weights.0'].cpu().numpy()
    # chemberta_feature_wt = model_state['weights.1'].cpu().numpy()
    # mordred_feature_wt = model_state['weights.2'].cpu().numpy()
    # signaturizer_feature_wt = model_state['weights.3'].cpu().numpy()
    # imagemol_feature_wt = model_state['weights.4'].cpu().numpy()
    # grover_feature_wt = model_state['weights.5'].cpu().numpy()


    # model_weights =  [mopac_feature_wt ,
    # chemberta_feature_wt,
    # mordred_feature_wt,
    # signaturizer_feature_wt ,
    # imagemol_feature_wt ,
    # grover_feature_wt ]




    return model, loss_train, loss_val, acc_train, acc_val, embeddings#,  model_weights#, model_state


def trainAE(model, dataset_name, embed_dim, train_loader, val_loader, epochs, verbose = False):
    
    # These are the parameters
    
    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.MSELoss()
    LEARNING_RATE = 5e-1
    WEIGHT_DECAY = 0
    OPTIMIZER = optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience = 5, verbose = True)

    loss_train = []
    loss_val = []

    acc_train = []
    acc_val = []

    alpha, beta, gamma = 0.33, 0.33, 0.33
    # alpha, beta = 0.5, 0.5

    for epoch in range(NUM_EPOCHS):

        st = time.time()
        model.train()
        # print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0 
        average_batch_loss_train = 0

        for i, data in enumerate(train_loader, 0):
            OPTIMIZER.zero_grad()
            
            k1, k2, k3, k4, k5, k6, labels = data

            k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            labels = labels.to(device)

            key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
            key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
            key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
            key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
            key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
            key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)

            key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])

            encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
            encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
            encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
            encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
            encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
            encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

            reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
            reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
            reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
            reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
            reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
            reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

            concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
            reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

            total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
            total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6
            # print('train',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
            total_loss_encoder = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)

            # print(device)
            total_loss_encoder.backward()
            OPTIMIZER.step()

            _loss = total_loss_encoder.item()

            batch_loss_train += _loss

            average_batch_loss_train = batch_loss_train / (i+1)

        loss_train.append(average_batch_loss_train)

        SCHEDULER.step(average_batch_loss_train)


        ## Validation
        batch_loss_val = 0
        avg_loss_val = 0

        # with torch.no_grad():
        #     model.eval()
        #     for i, data in enumerate(val_loader):
        #         k1, k2, k3, k4, k5, k6, labels = data
        #         k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
        #         labels = labels.to(device)

        #         key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
        #         key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
        #         key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
        #         key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
        #         key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
        #         key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)
                
        #         key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])
            
        #         encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
        #         encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
        #         encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
        #         encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
        #         encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
        #         encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

        #         reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
        #         reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
        #         reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
        #         reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
        #         reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
        #         reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

        #         concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
        #         reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

        #         total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
        #         total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6

        #         # print('val',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
        #         total_loss_encoder_val = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)
                
        #         _loss_val = total_loss_encoder_val.item()
        #         batch_loss_val += _loss_val
        #         avg_loss_val = batch_loss_val / (i+1)

        #     loss_val.append(avg_loss_val)
        if verbose:
            print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} time: {time.time() - st}')
            # print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')

    # print('Training FINISHED')
    print(f'Training AUTOENCODER finished Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')


    torch.save(model.state_dict(), f"{dataset_name}_cdi.pt")

    return model, loss_train, loss_val, acc_train, acc_val,  #model_weights#, model_state


def finetune_AE(model, user_embed_dim, choice, train_loader, epochs, verbose = False):
    
    # These are the parameters
    
    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.MSELoss()
    LEARNING_RATE = 5e-1
    WEIGHT_DECAY = 0
    OPTIMIZER = optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience = 5, verbose = True)

    loss_train = []
    loss_val = []

    acc_train = []
    acc_val = []


    for epoch in range(NUM_EPOCHS):

        st = time.time()
        model.train()
        # print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0 
        average_batch_loss_train = 0

        for i, data in enumerate(train_loader, 0):
            OPTIMIZER.zero_grad()
            
            k1, k2, k3, k4, k5, k6, labels = data
            k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            labels = labels.to(device)
            
            embed, embed_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])
            
            loss = LOSS_CRITERION(embed, embed_reconstruction)            
            loss.backward()
            OPTIMIZER.step()

            _loss = loss.item()
            batch_loss_train += _loss
            average_batch_loss_train = batch_loss_train / (i+1)
        
        loss_train.append(average_batch_loss_train)
        SCHEDULER.step(average_batch_loss_train)

        ## Validation
        batch_loss_val = 0
        avg_loss_val = 0

        # with torch.no_grad():
        #     model.eval()
        #     for i, data in enumerate(val_loader):
        #         k1, k2, k3, k4, k5, k6, labels = data
        #         k1, k2, k3, k4, k5, k6= k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
        #         labels = labels.to(device)

        #         key_1 = torch.cat([k2, k3, k4, k5, k6], dim=1)
        #         key_2 = torch.cat([k1, k3, k4, k5, k6], dim=1)
        #         key_3 = torch.cat([k1, k2, k4, k5, k6], dim=1)
        #         key_4 = torch.cat([k1, k2, k3, k5, k6], dim=1)
        #         key_5 = torch.cat([k1, k2, k3, k4, k6], dim=1)
        #         key_6 = torch.cat([k1, k2, k3, k4, k5], dim=1)
                
        #         key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc, key_1_reconstruction, key_2_reconstruction, key_3_reconstruction, key_4_reconstruction, key_5_reconstruction, key_6_reconstruction, _, concat_reconstruction = model.forward([k1, k2, k3, k4, k5, k6])
            
        #         encoding_loss1 = LOSS_CRITERION(key_1_enc, k1)
        #         encoding_loss2 = LOSS_CRITERION(key_2_enc, k2)
        #         encoding_loss3 = LOSS_CRITERION(key_3_enc, k3)
        #         encoding_loss4 = LOSS_CRITERION(key_4_enc, k4)
        #         encoding_loss5 = LOSS_CRITERION(key_5_enc, k5)
        #         encoding_loss6 = LOSS_CRITERION(key_6_enc, k6)

        #         reconstruction_loss1 = LOSS_CRITERION(key_1_reconstruction, key_1)
        #         reconstruction_loss2 = LOSS_CRITERION(key_2_reconstruction, key_2)
        #         reconstruction_loss3 = LOSS_CRITERION(key_3_reconstruction, key_3)
        #         reconstruction_loss4 = LOSS_CRITERION(key_4_reconstruction, key_4)
        #         reconstruction_loss5 = LOSS_CRITERION(key_5_reconstruction, key_5)
        #         reconstruction_loss6 = LOSS_CRITERION(key_6_reconstruction, key_6)

        #         concat_key = torch.cat([key_1_enc, key_2_enc, key_3_enc, key_4_enc, key_5_enc, key_6_enc], dim=1)
        #         reconstruction_loss_concat = LOSS_CRITERION(concat_key, concat_reconstruction)

        #         total_encoding_loss = encoding_loss1 + encoding_loss2 + encoding_loss3 + encoding_loss4 + encoding_loss5 + encoding_loss6
        #         total_reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3 + reconstruction_loss4 + reconstruction_loss5 + reconstruction_loss6

        #         # print('val',total_encoding_loss, total_reconstruction_loss, reconstruction_loss_concat)
        #         total_loss_encoder_val = (alpha * total_encoding_loss / 6) + (beta * total_reconstruction_loss / 6) + (gamma * reconstruction_loss_concat)
                
        #         _loss_val = total_loss_encoder_val.item()
        #         batch_loss_val += _loss_val
        #         avg_loss_val = batch_loss_val / (i+1)

        #     loss_val.append(avg_loss_val)

        if verbose:
            print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} time: {time.time() - st}')
            # print(f'Epoch: {epoch + 1} Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')

    # print('Training FINISHED')
    print(f'Training AUTOENCODER finished Train loss: {average_batch_loss_train} ')
    # print(f'Training AUTOENCODER finished Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')



    return model, loss_train, loss_val, acc_train, acc_val


def trainModel(model, embed_dim, train_loader, val_loader, epochs, verbose):
    
    # These are the parameters
    
    NUM_EPOCHS = epochs
    LOSS_CRITERION = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-3
    # LEARNING_RATE = 1e-2
    # LEARNING_RATE = 1e-1
    WEIGHT_DECAY = 0
    OPTIMIZER = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, patience = 3, verbose = verbose)


    loss_train = []
    loss_val = []

    acc_train = []
    acc_val = []

    best_val_acc = -1
    best_epoch = 1

    for epoch in range(NUM_EPOCHS):
        model.train()
        # print(f'Epoch: {epoch + 1}')

        batch_loss_train = 0 
        average_batch_loss_train = 0

        train_correct = 0
        train_samples = 0

        for i, data in enumerate(train_loader, 0):
            OPTIMIZER.zero_grad()
            
            # k1, k2, k3, k4, k5, k6, labels = data
            # k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            # labels = labels.to(device)

            x, labels = data
            x, labels = x.to(device), labels.to(device)
            
            outputs = model.forward(x)
            # outputs = model.forward([k1, k2, k3, k4, k5, k6])

            loss = LOSS_CRITERION(outputs, labels)

            loss.backward()
            OPTIMIZER.step()
            
            outputs = nn.Softmax(dim = 1)(outputs)
            _, prediction = torch.max(outputs, dim = 1)

            train_correct += (prediction == labels).sum().item()
            train_samples += len(outputs)

            _loss = loss.item()
            batch_loss_train += _loss
            average_batch_loss_train = batch_loss_train / (i+1)
        
        loss_train.append(average_batch_loss_train)
        SCHEDULER.step(average_batch_loss_train)

        train_acc = (train_correct / train_samples) * 100
        acc_train.append(train_acc)

        if verbose: print(f'Epoch: {epoch + 1} Training loss: {round(average_batch_loss_train, 3)} Training acc: {round(train_acc, 3)} ')
        file_path = f"../weights/trainCLS_1/{embed_dim}"
        makeDir(file_path)
        torch.save(model.state_dict(), file_path + f'{epoch + 1}.pt')
        

        ## Validation
        batch_loss_val = 0
        avg_loss_val = 0

        val_correct = 0
        val_samples = 0
        avg_loss_val = 0

        with torch.no_grad():
            model.eval()
            for i, data in enumerate(val_loader):
                # k1, k2, k3, k4, k5, k6, labels = data
                # k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
                # labels = labels.to(device)

                x, labels = data
                x, labels = x.to(device), labels.to(device)
                
                outputs = model.forward(x)
                # outputs = model.forward([k1, k2, k3, k4, k5, k6])
                loss = LOSS_CRITERION(outputs, labels)

                _loss_val = loss.item()
                batch_loss_val += _loss_val
                avg_loss_val = batch_loss_val / (i+1)

                outputs = nn.Softmax(dim = 1)(outputs)
                _, prediction = torch.max(outputs, dim = 1)

                val_correct += (prediction == labels).sum().item()
                val_samples += len(outputs)

            loss_val.append(avg_loss_val)
            
            val_acc = (val_correct / val_samples) * 100
            acc_val.append(val_acc)
            
            if verbose: print(f'Epoch: {epoch + 1} val loss: {round(avg_loss_val, 3)}, val acc: {round(val_acc, 3)}')

            if round(val_acc, 3) >= best_val_acc:
                best_val_acc = round(val_acc, 3)
                best_epoch = epoch+1

    # print('Training FINISHED')
    print(f'Training CLASSIFIER finished Train loss: {average_batch_loss_train} val loss: {avg_loss_val}')

    # torch.save(model.state_dict(), f"../weights/{epoch + 1}.pt")

    return model, best_epoch, loss_train, loss_val, acc_train, acc_val

def testModel(model, test_loader, verbose):
    test_correct = 0
    test_samples = 0

    predicted_label = []
    actual_label = []
    predicted_proba = []
    test_ids = []

    with torch.no_grad():
        LOSS_CRITERION = nn.CrossEntropyLoss()
        for data in test_loader:
            # k1, k2, k3, k4, k5, k6, labels = data
            # k1, k2, k3, k4, k5, k6 = k1.to(device), k2.to(device), k3.to(device), k4.to(device), k5.to(device), k6.to(device)
            # labels = labels.to(device)

            x, labels = data
            x, labels = x.to(device), labels.to(device)
            
            outputs = model.forward(x)
            # outputs = model.forward(k1, k2, k3, k4, k5, k6)
            loss = LOSS_CRITERION(outputs, labels)

            outputs = nn.Softmax(dim = 1)(outputs)
            _, prediction = torch.max(outputs, dim = 1)
            
            test_correct += (prediction == labels).sum().item()
            test_samples += len(outputs)

            predicted_label.extend(prediction.detach().cpu().numpy())
            actual_label.extend(labels.detach().cpu().numpy())
            predicted_proba.extend(outputs[:, 1].detach().cpu().numpy())
            
    test_acc = (test_correct / test_samples) * 100

    actual_lbl = actual_label
    predicted_lbl = predicted_label
    pred_prob = predicted_proba


    if verbose: 
        print(f'Accuracy on test set: {round(test_acc, 4)} %')
        f1 = f1_score(actual_lbl, predicted_lbl, average = 'macro')
        print(f'F1 score on test set: {f1}')
        print(f'Balanced Accuracy on test set: {round(balanced_accuracy_score(actual_lbl, predicted_lbl) * 100, 4)} %')
        print(f'ROC AUC score: ', roc_auc_score(actual_lbl, predicted_proba))
        print(classification_report(actual_lbl, predicted_lbl))
        print(confusion_matrix(actual_lbl, predicted_lbl))


    return actual_lbl, predicted_lbl, pred_prob


def writeIntoFile(file_path, file_name, ids, y_pred, y_actual, y_p1):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(file_path + file_name, mode='w', newline='') as file:
            writer = csv.writer(file) 
            writer.writerow(['ID', 'Predicted Label', 'Actual Label', 'P1'])
            for row in range(len(ids)):
                writer.writerow([ids[row], y_pred[row], y_actual[row], y_p1[row]])

def plotROC(train_actual_lbl, train_pred_prob, test_actual_lbl, test_pred_prob, choice):
    train_fpr, train_tpr, _ = roc_curve(train_actual_lbl, train_pred_prob)
    train_roc_auc = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, _ = roc_curve(test_actual_lbl, test_pred_prob)
    test_roc_auc = auc(test_fpr, test_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training ROC curve (AUC = {train_roc_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='blue', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../figures/Evolf ROC {choice}.pdf', format="pdf", bbox_inches="tight") 
    plt.show()

def plotROC(train_actual_lbl, train_pred_prob, test_actual_lbl, test_pred_prob, choice):
    train_fpr, train_tpr, _ = roc_curve(train_actual_lbl, train_pred_prob)
    train_roc_auc = auc(train_fpr, train_tpr)

    test_fpr, test_tpr, _ = roc_curve(test_actual_lbl, test_pred_prob)
    test_roc_auc = auc(test_fpr, test_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(train_fpr, train_tpr, color='darkorange', lw=2, label=f'Training ROC curve (AUC = {train_roc_auc:.2f})')
    plt.plot(test_fpr, test_tpr, color='blue', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'../figures/Comb ROC {choice}.pdf', format="pdf", bbox_inches="tight") 
    plt.show()

def pickleDump(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + file_name, 'wb') as f:
        pickle.dump(content, f)

def pickleRead(file_path, file_name):
    if (not os.path.exists(file_path)) or ( not os.path.exists(file_path + file_name)):
        return None
    with open(file_path + file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def makeDir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def writeIntoFile(file_path, file_name, content):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(len(content)):
            writer.writerow([content[row]])




