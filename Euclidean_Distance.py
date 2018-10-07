# Intra class distance function
#   database_features - numpy array of extracted features (with class label as final column)
def intra_class_distance(database_features):
    output_file = open("intra_class_distance.txt","w")

    labels = database_features[:,-1]
    database_features = database_features[:,:-1]

    n,d = database_features.shape 

    for i in range(n):
        for j in range(n):
            if j > i and labels[i] == labels[j]:
                dist = cv2.norm(database_features[i,:],database_features[j,:])
                output_file.write(str(dist) + "\n")
                output_file.flush()

    output_file.close()
    return

# Inter class distance function
#   database_features - numpy array of extracted features (with class label as final column)
def inter_class_distance(database_features):
    output_file = open("inter_class_distance.txt","w")

    labels = database_features[:,-1]
    database_features = database_features[:,:-1]

    n,d = database_features.shape 

    for i in range(n):
        for j in range(n):
            if i < j and labels[i] != labels[j]:
                dist = cv2.norm(database_features[i,:],database_features[j,:])
                output_file.write(str(dist) + "\n")
                output_file.flush()

    output_file.close()
    return
