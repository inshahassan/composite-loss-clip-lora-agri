# Corrected IP102 taxonomy (class index -> order, family), built from the class names in ip102.yaml.
# Replaces the earlier index-range assumption (e.g. "classes 0-17 = Hemiptera"), which did not
# match the real IP102 class order (IP102 classes are grouped by crop, not by insect order).
# Entries marked "# CHECK" are the least certain identifications.
_T = [
    ("Lepidoptera", "Crambidae"),        # 0 rice leaf roller
    ("Lepidoptera", "Hesperiidae"),      # 1 rice leaf caterpillar
    ("Diptera", "Ephydridae"),           # 2 paddy stem maggot
    ("Lepidoptera", "Crambidae"),        # 3 asiatic rice borer
    ("Lepidoptera", "Crambidae"),        # 4 yellow rice borer
    ("Diptera", "Cecidomyiidae"),        # 5 rice gall midge
    ("Diptera", "Chloropidae"),          # 6 Rice Stemfly
    ("Hemiptera", "Delphacidae"),        # 7 brown plant hopper
    ("Hemiptera", "Delphacidae"),        # 8 white backed plant hopper
    ("Hemiptera", "Delphacidae"),        # 9 small brown plant hopper
    ("Coleoptera", "Curculionidae"),     # 10 rice water weevil
    ("Hemiptera", "Cicadellidae"),       # 11 rice leafhopper
    ("Thysanoptera", "Thripidae"),       # 12 grain spreader thrips
    ("Hemiptera", "Pentatomidae"),       # 13 rice shell pest  # CHECK
    ("Coleoptera", "Scarabaeidae"),      # 14 grub
    ("Orthoptera", "Gryllotalpidae"),    # 15 mole cricket
    ("Coleoptera", "Elateridae"),        # 16 wireworm
    ("Lepidoptera", "Erebidae"),         # 17 white margined moth  # CHECK
    ("Lepidoptera", "Noctuidae"),        # 18 black cutworm
    ("Lepidoptera", "Noctuidae"),        # 19 large cutworm
    ("Lepidoptera", "Noctuidae"),        # 20 yellow cutworm
    ("Acari", "Tetranychidae"),          # 21 red spider
    ("Lepidoptera", "Crambidae"),        # 22 corn borer
    ("Lepidoptera", "Noctuidae"),        # 23 army worm
    ("Hemiptera", "Aphididae"),          # 24 aphids
    ("Coleoptera", "Scarabaeidae"),      # 25 Potosia brevitarsis
    ("Lepidoptera", "Crambidae"),        # 26 peach borer
    ("Hemiptera", "Aphididae"),          # 27 english grain aphid
    ("Hemiptera", "Aphididae"),          # 28 green bug
    ("Hemiptera", "Aphididae"),          # 29 bird cherry-oat aphid
    ("Diptera", "Cecidomyiidae"),        # 30 wheat blossom midge
    ("Acari", "Penthaleidae"),           # 31 penthaleus major
    ("Acari", "Tetranychidae"),          # 32 longlegged spider mite
    ("Thysanoptera", "Phlaeothripidae"), # 33 wheat phloeothrips
    ("Hymenoptera", "Tenthredinidae"),   # 34 wheat sawfly
    ("Diptera", "Agromyzidae"),          # 35 cerodonta denticornis
    ("Diptera", "Anthomyiidae"),         # 36 beet fly
    ("Coleoptera", "Chrysomelidae"),     # 37 flea beetle
    ("Lepidoptera", "Noctuidae"),        # 38 cabbage army worm
    ("Lepidoptera", "Noctuidae"),        # 39 beet army worm
    ("Diptera", "Anthomyiidae"),         # 40 Beet spot flies
    ("Lepidoptera", "Crambidae"),        # 41 meadow moth
    ("Coleoptera", "Curculionidae"),     # 42 beet weevil
    ("Coleoptera", "Scarabaeidae"),      # 43 Serica orientalis
    ("Coleoptera", "Curculionidae"),     # 44 alfalfa weevil
    ("Lepidoptera", "Noctuidae"),        # 45 flax budworm
    ("Hemiptera", "Miridae"),            # 46 alfalfa plant bug
    ("Hemiptera", "Miridae"),            # 47 tarnished plant bug
    ("Orthoptera", "Acrididae"),         # 48 Locustoidea
    ("Coleoptera", "Meloidae"),          # 49 lytta polita
    ("Coleoptera", "Meloidae"),          # 50 legume blister beetle
    ("Coleoptera", "Meloidae"),          # 51 blister beetle
    ("Hemiptera", "Aphididae"),          # 52 Therioaphis maculata
    ("Thysanoptera", "Thripidae"),       # 53 odontothrips loti
    ("Thysanoptera", "Thripidae"),       # 54 Thrips
    ("Hymenoptera", "Eurytomidae"),      # 55 alfalfa seed chalcid
    ("Lepidoptera", "Pieridae"),         # 56 Pieris canidia
    ("Hemiptera", "Miridae"),            # 57 Apolygus lucorum
    ("Lepidoptera", "Limacodidae"),      # 58 Limacodidae
    ("Hemiptera", "Phylloxeridae"),      # 59 Viteus vitifoliae
    ("Acari", "Eriophyidae"),            # 60 Colomerus vitis
    ("Acari", "Tenuipalpidae"),          # 61 Brevipalpus lewisi
    ("Coleoptera", "Chrysomelidae"),     # 62 oides decempunctata
    ("Acari", "Tarsonemidae"),           # 63 Polyphagotarsonemus latus
    ("Hemiptera", "Pseudococcidae"),     # 64 Pseudococcus comstocki
    ("Lepidoptera", "Sesiidae"),         # 65 parathrene regalis
    ("Lepidoptera", "Sphingidae"),       # 66 Ampelophaga
    ("Hemiptera", "Fulgoridae"),         # 67 Lycorma delicatula
    ("Coleoptera", "Cerambycidae"),      # 68 Xylotrechus
    ("Hemiptera", "Cicadellidae"),       # 69 Cicadella viridis
    ("Hemiptera", "Miridae"),            # 70 Miridae
    ("Hemiptera", "Aleyrodidae"),        # 71 Trialeurodes vaporariorum
    ("Hemiptera", "Cicadellidae"),       # 72 Erythroneura apicalis
    ("Lepidoptera", "Papilionidae"),     # 73 Papilio xuthus
    ("Acari", "Tetranychidae"),          # 74 Panonychus citri
    ("Acari", "Eriophyidae"),            # 75 Phyllocoptes oleiverus
    ("Hemiptera", "Monophlebidae"),      # 76 Icerya purchasi
    ("Hemiptera", "Diaspididae"),        # 77 Unaspis yanonensis
    ("Hemiptera", "Coccidae"),           # 78 Ceroplastes rubens
    ("Hemiptera", "Diaspididae"),        # 79 Chrysomphalus aonidum
    ("Hemiptera", "Diaspididae"),        # 80 Parlatoria zizyphus
    ("Hemiptera", "Pseudococcidae"),     # 81 Nipaecoccus vastalor
    ("Hemiptera", "Aleyrodidae"),        # 82 Aleurocanthus spiniferus
    ("Diptera", "Tephritidae"),          # 83 Bactrocera minax
    ("Diptera", "Tephritidae"),          # 84 Dacus dorsalis
    ("Diptera", "Tephritidae"),          # 85 Bactrocera tsuneonis
    ("Lepidoptera", "Noctuidae"),        # 86 Prodenia litura
    ("Lepidoptera", "Erebidae"),         # 87 Adris tyrannus  # CHECK
    ("Lepidoptera", "Gracillariidae"),   # 88 Phyllocnistis citrella
    ("Hemiptera", "Aphididae"),          # 89 Toxoptera citricidus
    ("Hemiptera", "Aphididae"),          # 90 Toxoptera aurantii
    ("Hemiptera", "Aphididae"),          # 91 Aphis citricola
    ("Thysanoptera", "Thripidae"),       # 92 Scirtothrips dorsalis
    ("Diptera", "Cecidomyiidae"),        # 93 Dasineura sp
    ("Hemiptera", "Flatidae"),           # 94 Lawana imitata
    ("Hemiptera", "Flatidae"),           # 95 Salurnis marginella
    ("Coleoptera", "Attelabidae"),       # 96 Deporaus marginatus
    ("Lepidoptera", "Noctuidae"),        # 97 Chlumetia transversa  # CHECK
    ("Hemiptera", "Cicadellidae"),       # 98 Mango flat beak leafhopper
    ("Coleoptera", "Cerambycidae"),      # 99 Rhytidodera bowrinii
    ("Coleoptera", "Curculionidae"),     # 100 Sternochetus frigidus
    ("Hemiptera", "Cicadellidae"),       # 101 Cicadellidae
]

assert len(_T) == 102

_PREFIX = {"Lepidoptera": "lep", "Hemiptera": "hem", "Coleoptera": "col", "Diptera": "dip",
           "Thysanoptera": "thy", "Acari": "aca", "Orthoptera": "ort", "Hymenoptera": "hym"}

IP102_TAXONOMY = {
    c: {"order": o, "family": f, "genus": f"{_PREFIX[o]}_{c}"}
    for c, (o, f) in enumerate(_T)
}
