# CAV_LEMEUR
Image Super-Resolution as Sparse Representation of Raw Image Patches
Ne marche pas sous Windows à cause de cv::glob, erreur d'accès à un vecteur vide.

cd build ; make : to build the project;

_Main.cpp : 
	PATCH_SIZE x : taille des patchs
	void constructionDictionnaires() : préciser le chemin de la base d'apprentissage
	void superResolution() : préciser le chemin de l'image d'entrée
				index = fonction_de_scoring

./SR_LEMEUR : exécuter le projet -> suivre les instructions.


