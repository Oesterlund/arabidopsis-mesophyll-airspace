# Annotering af celler

`images/`
  Billeder der er nedskaleret med en faktor 4 og beskåret til en kasse rundt om bladet.

`segmentations/`
  `epidermis_2d_model/`
    Segmenteringer af epidermis fra 2d model
	
`annotations`
  `palisade_mesophyll/`
    Gem annoteringer af palisade mesophyll her.
   
`scripts`
  `annotator.py`
    Script til at annotere med. Hvis du skriver
		`python annotator.py --help`
	bliver der printet hjælp. Den kræver at der er `napari`, `numpy` og `nibabel`
	

Der er to måder at bruge `annotator.py` på. 

1. Når du starter en annotering kan du bruge segmenterings masker som udgangspunkt for annoteringen. 

		python3 scripts/annotator.py <billede> <ud-fil> --background-mask <segmentering>

Hvis vi f.eks annotere spongy mesophyll og gerne vil starte med epidermis segmenteringer som baggrund kan vi gøre følgende

		python3 scripts/annotator.py images/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz annotations/spongy_mesophyll/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz --background-mask segmentations/epidermis_2d_model/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz 
	
Det vil give følgende output fordi masken har zero padding der skal fjernes for at den overlapper med billedet.

	Cropping background mask to fit image
	Mask shape (540, 448, 640)
	Image shape (540, 340, 640)
	Crop (slice(0, 540, None), slice(54, 394, None), slice(0, 640, None))

Hvis der bliver croppet, så tjek masken. Er den dårlig så prøv igen uden.
Du kan bruge flere masker som baggund til at starte med, hvis der f.eks også var en segmentering af luft.

Når napari vinduet er åbent kan du bare annotere. Annoteringen bliver gemt når du lukker. Start med at lave lidt, lukke vinduet og så tjekke at det virker. Du kan bruge `viewer.py` til at tjekke.

	
	python3 scripts/viewer.py <billede> [<maske>*]
	
F.eks

	python3 scripts/viewer.py images/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz annotations/spongy_mesophyll/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz


Hvis du vil tegne videre på en annotering skal du gøre følgende

	python3 scripts/annotator.py <billede> <ud-fil> --annotation <den-gamle-annotering>

F.eks
		
	python3 scripts/annotator.py images/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz annotations/spongy_mesophyll/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz --annotation annotations/spongy_mesophyll/140_RIC_w6_p2_l8m_zoomed-0.25.nii.gz


Du kan godt give annoteringsfilen et nyt navn hvis du ikke vil overskrive den.
