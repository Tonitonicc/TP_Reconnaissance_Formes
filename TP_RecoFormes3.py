import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
class FuzzyCMeans:
    def __init__(self, nbr_clusters=2,m=2,nbr_iter_max=100,seuil=0.01, seed=42):
        self.nbr_clusters=nbr_clusters
        self.m= m
        self.nbr_iter_max=nbr_iter_max
        self.seuil=seuil
        self.seed=seed
        self.centres= None
        self.matrice_u= None
        self.historique=  []
        
    def initialiser_matrice(self,nbr_pts):
        np.random.seed(self.seed)
        mat=np.random.rand(nbr_pts,self.nbr_clusters)
        for i in range(nbr_pts):
            somme=0
            for j in range(self.nbr_clusters):
                somme +=mat[i,j]
            for j in range(self.nbr_clusters):
                mat[i,j]=mat[i,j]/  somme
        return mat
    def calculer_centres(self,data,matrice_u):
        centres=np.zeros((self.nbr_clusters,data.shape[1]))
        for j in range(self.nbr_clusters):
            numerateur = 0
            denominateur = 0
            for i in range(data.shape[0]):
                poids = matrice_u[i, j] ** self.m
                numerateur += poids * data[i]
                denominateur += poids
            centres[j] = numerateur / denominateur
        return centres
    def calculer_distances(self,data,centres):
        distance=np.zeros((data.shape[0],self.nbr_clusters))
        for i in range(data.shape[0]):
            for j in range(self.nbr_clusters):
                differrence=data[i]-centres[j]
                distance[i,j]=np.sqrt(np.sum(differrence*differrence))
        return distance
    
    def mettre_a_jour_matrice(self,distance):
        puissance=2/(self.m-1)
        nouvelle_mat=np.zeros((distance.shape[0],self.nbr_clusters))
        
        for i in range(distance.shape[0]):
            for j in range(self.nbr_clusters):
                if distance[i,j]<0.0000000001:
                    nouvelle_mat[i,j]=1.0
                    for k in range(self.nbr_clusters):
                        if k !=j:
                            nouvelle_mat[i,k]=0.0
                    break
                else:
                    somme=0
                    for k in range(self.nbr_clusters):
                        if distance[i,k]<0.0000000001:
                            somme=999999999
                            break
                        rapport=distance[i,j]/distance[i,k]
                        somme +=rapport**puissance
                    
                    if somme>999999998:
                        nouvelle_mat[i,j]=0.0
                    else:
                        nouvelle_mat[i,j]=1.0/ somme
        return nouvelle_mat
    
    def fit(self,data):
        nb_points=data.shape[0]
        self.matrice_u=self.initialiser_matrice(nb_points)
        for iteration in range(self.nbr_iter_max):
            matrice_ancienne=self.matrice_u.copy()
            self.centres=self.calculer_centres(data,self.matrice_u)            
            distance=self.calculer_distances(data,self.centres) 
            self.matrice_u=self.mettre_a_jour_matrice(distance)
            changement=0
            for i in range(self.matrice_u.shape[0]):
                for j in range(self.matrice_u.shape[1]):
                    difference=self.matrice_u[i,j]-matrice_ancienne[i,j]
                    changement +=difference*difference
            changement=np.sqrt(changement)
            self.historique.append(changement)
            print(f"itération : {iteration + 1}/{self.nbr_iter_max} ett changement:{changement:.6f}")
            if changement<self.seuil:
                print(f"Convergence a l'iteration{iteration+1}")
                break
        return self
    def predict(self):
        labels=np.zeros(self.matrice_u.shape[0],dtype=int)
        for i in range(self.matrice_u.shape[0]):
            max_val= -1
            max_idx= 0
            for j in range(self.matrice_u.shape[1]):
                if self.matrice_u[i,j]>max_val:
                    max_val=self.matrice_u[i, j]
                    max_idx=j
            labels[i]=max_idx
        return labels
    def get_membership_matrix(self):
        return self.matrice_u
def charger_image(chemin_image):
    img=Image.open(chemin_image)
    if img.mode != 'L':
        img=img.convert('L')
    tableau_image=np.array(img)
    h, w=tableau_image.shape
    if h>500 or w>500:
        ratio=min(500/h,500/w)
        nouveau_h,nouveau_w = int(h*ratio),int(w*ratio)
        img=img.resize((nouveau_w, nouveau_h),Image.LANCZOS)
        tableau_image=np.array(img)
    forme=tableau_image.shape
    tableau_normalise=tableau_image.astype(np.float64)/255
    data =tableau_normalise.reshape(-1,1)
    
    return tableau_image,data,forme
def afficher_resultats(img_originale,mat_u,labels,forme,fcm,  dossier_sortie,k):
    h,w=forme
    img_labels=labels.reshape(h,w)
    nbr_clusters=mat_u.shape[1]
    fig, axes=plt.subplots(2,nbr_clusters+1,figsize=(15,8))
    axes[0, 0].imshow(img_originale,cmap='gray')
    axes[0, 0].set_title('Image Originale')
    axes[0, 0].axis('off')
    
    for j in range(nbr_clusters):
        appartenance=mat_u[:,j].reshape(h,w)
        im = axes[0,j+1].imshow(appartenance, cmap='hot',vmin=0,vmax=1)
        axes[0,j+1].axis('off')
        plt.colorbar(im,ax=axes[0,j+1],fraction=0.046)
    
    axes[1, 0].imshow(img_labels, cmap='viridis')
    axes[1, 0].set_title('Segmentation Final')
    axes[1, 0].axis('off')
    
    for j in range(nbr_clusters):
        img_cluster=img_originale.copy()
        masque_cluster=(img_labels==j)
        img_cluster[~masque_cluster]=0
        axes[1,j+1].imshow(img_cluster, cmap='gray')
        axes[1,j+1].set_title(f'Pixels du Cluster {j+1}')
        axes[1,j+1].axis('off')
    plt.tight_layout()
    plt.savefig(f'{dossier_sortie}/segmentation_K{k}.png', dpi=300, bbox_inches='tight')
    plt.show()
def afficher_convergence(resultats_fcm,dossier_sortie='results'):
    plt.figure(figsize=(12,6))
    couleurs = ["#002AFF", "#E42525", "#FF7C02"]

    for i,(k,fcm) in enumerate(resultats_fcm.items()):
        plt.plot(fcm.historique, marker='o',linewidth=2,label=f'K={k}', 
                color=couleurs[i],markersize= 6,alpha=0.8)
    plt.xlabel('Itération', fontsize=13, fontweight='bold')
    plt.ylabel('Changement de U', fontsize=13, fontweight='bold')
    plt.title('Comparaison de la convergence pour différentes valeurs de K')
    #plt.legend(loc='upper right')
    plt.grid(True,alpha=0.3, linestyle=' --')
    plt.tight_layout()
    plt.savefig(f'{dossier_sortie}/convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def comparer_segmentations(img_originale,resultats_k2,resultats_k3,forme,dossier_sortie='results'):
    h,w=forme
    etiq_k2=resultats_k2['labels'].reshape(h,w)

    etiq_k3=resultats_k3['labels'].reshape(h,w)
    fig, axes=plt.subplots(2,3,figsize=(18,12))
    
    axes[0, 0].imshow(img_originale, cmap='gray')
    axes[0, 0].set_title('Image OG')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(etiq_k2, cmap='viridis')
    axes[0, 1].set_title('Segmentationk=2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(etiq_k3, cmap='viridis')
    axes[0, 2].set_title('Segmentation k=3')
    axes[0, 2].axis('off')
    
    mat_u_k2=resultats_k2['U']
    mat_u_k3=resultats_k3['U']
    pls_lumineux_k2=0
    max_val=resultats_k2['centers'][0]
    for i in range(len(resultats_k2['centers'])):
        if resultats_k2['centers'][i]>max_val:
            max_val=resultats_k2['centers'][i]
            pls_lumineux_k2=i
    appart_k2=mat_u_k2[:,pls_lumineux_k2].reshape(h, w)
    pls_lumineux_k3 = 0
    max_val=resultats_k3['centers'][0]
    for i in range(len(resultats_k3['centers'])):
        if resultats_k3['centers'][i]>max_val:
            max_val=resultats_k3['centers'][i]
            pls_lumineux_k3 =i
    appart_k3=mat_u_k3[:,pls_lumineux_k3].reshape(h,w)
    axes[1, 0].axis('off')
    im1=axes[1,1].imshow(appart_k2, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Appartenance k=2')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    im2 = axes[1, 2].imshow(appart_k3, cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title('Appartenance k=3')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    plt.suptitle('Comparaison des segmentations pour 2et 3', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{dossier_sortie}/comparison_K2_vs_K3.png', dpi=300, bbox_inches='tight')
    plt.show()
def lancer_experience(img_orig,data,forme,k,m,seuil,nbr_iter,dossier):
    fcm=FuzzyCMeans(nbr_clusters=k,m=m,nbr_iter_max=nbr_iter,seuil=seuil,seed=42)
    fcm.fit(data)
    etiq=fcm.predict()
    mat_u=fcm.get_membership_matrix()
    for i, centre in enumerate(fcm.centres):
        print(f"cluster {i+1}: {centre[0]:.4f}")
    
    print(f"Répartition des pixels (K={k}):")
    for i in range(k):
        compteur=0
        for j in range(len(etiq)):
            if etiq[j] == i:
                compteur +=1
        pourcentage=(compteur/len(etiq))*100
        print(f" Cluster {i+1}: {compteur}pixels ({pourcentage:.2f}%)")
    afficher_resultats(img_orig,mat_u,etiq,forme,fcm,dossier,k) 
    return {
        'fcm': fcm,
        'labels': etiq,
        'U': mat_u,
        'centers': fcm.centres
    }
def main():    
    chemin= "C:/Users/antoi/OneDrive/Bureau/UPC/M2/Reconnaissance des formes/milky-way-nvg.jpg"
    m=2.0
    seuil=0.01
    nbr_iter=100
    dossier='results'
    print(f"  - Itérations max: {nbr_iter}")
    os.makedirs(dossier,exist_ok=True)
    img_orig,data,forme=charger_image(chemin)
    print(f"Img chargée:{forme[0]}x{forme[1]} pixels({data.shape[0]} points)")
    resultats_k2 = lancer_experience(img_orig, data, forme,k=2,m=m,seuil= seuil,nbr_iter=nbr_iter,dossier =dossier)
    resultats_k3 = lancer_experience(img_orig, data, forme,k=3,m=m,seuil= seuil,nbr_iter=nbr_iter,dossier =dossier)
    dict_fcm = {
        2: resultats_k2['fcm'],
        3: resultats_k3['fcm']
    }
    afficher_convergence(dict_fcm,dossier)
    comparer_segmentations(img_orig,resultats_k2,resultats_k3,forme,dossier)
    
if __name__ == "__main__":
    main()