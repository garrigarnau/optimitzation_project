# **Optimització d'Entrenament de Xarxes Neuronals amb MNIST**

Aquest projecte té com a objectiu investigar i comparar l'entrenament bàsic i distribuït de models de xarxes neuronals convolucionals (`SimpleConvNet`) utilitzant el dataset MNIST, que conté imatges de dígits escrits a mà. Inclou implementacions per entrenar un model en mode bàsic i distribuït, amb mètriques per analitzar el rendiment, la pèrdua i la precisió.

## **Taula de continguts**
1. [Introducció](#introducció)
2. [Requisits](#requisits)
3. [Instal·lació](#instal·lació)
4. [Estructura del projecte](#estructura-del-projecte)
5. [Com utilitzar-lo](#com-utilitzar-lo)
    - [Entrenament bàsic](#entrenament-bàsic)
    - [Entrenament distribuït](#entrenament-distribuït)
6. [Resultats](#resultats)
7. [Futur desenvolupament](#futur-desenvolupament)
8. [Contribució](#contribució)
9. [Llicència](#llicència)

---

## **Introducció**
Aquest projecte busca explorar l'impacte de l'entrenament distribuït en xarxes neuronals. S'han creat dues versions:
- **Entrenament bàsic**: Un entrenament tradicional en un únic procés.
- **Entrenament distribuït**: Divisió del dataset en subconjunts per entrenar-los en paral·lel en diferents processos.

Això permet comparar el rendiment en termes de temps, precisió i pèrdua durant l'entrenament.

---

## **Requisits**
Abans de començar, assegura't que tens instal·lades les següents eines i biblioteques:
- **Python 3.8+**
- Llibreries Python:
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `numpy`
- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/) (es descarrega automàticament).

---

## **Instal·lació**
Segueix aquests passos per configurar el projecte localment:

1. Clona aquest repositori:
   ```bash
   git clone https://github.com/nom_usuari/nom_repositori.git
   ```
2. Entra al directori del projecte:
   ```bash
   cd nom_repositori
   ```
3. Instala les dependències necessàries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Estructura del projecte**
```plaintext
nom_repositori/
│
├── train_basic.py            # Script per a l'entrenament bàsic
├── train_distributed.py      # Script per a l'entrenament distribuït
├── results/                  # Directori per guardar resultats (models i gràfiques)
├── README.md                 # Documentació del projecte
├── requirements.txt          # Llista de dependències
└── data/                     # Directori on es descarrega el dataset MNIST
```

---

## **Com utilitzar-lo**

### **Entrenament bàsic**
1. Executa el fitxer `train_basic.py` per entrenar el model de manera bàsica:
   ```bash
   python train_basic.py
   ```
2. Els resultats es guardaran al directori `results/`:
   - Model entrenat: `model_basic.pth`
   - Gràfiques de pèrdua i precisió: `training_results.png`

### **Entrenament distribuït**
1. Executa el fitxer `train_distributed.py` per entrenar el model en mode distribuït:
   ```bash
   python train_distributed.py
   ```
2. Els resultats es guardaran al directori `results/`:
   - Gràfiques dels processos: `distributed_training_results.png`
   - Resultats complets: `distributed_results.pt`

---

## **Resultats**
### Entrenament bàsic:
- Pèrdua final i precisió del model visualitzades a `training_results.png`.

### Entrenament distribuït:
- Comparació del temps d'entrenament, pèrdues i precisió per cada procés visualitzada a `distributed_training_results.png`.

---

## **Futur desenvolupament**
Algunes idees per continuar millorant el projecte:
- Afegir suport per a entrenament en GPU per accelerar els processos.
- Escalar l'entrenament distribuït a múltiples màquines.
- Implementar altres arquitectures de xarxes neuronals.
- Realitzar experiments amb datasets més complexos.

---

## **Contribució**
Les contribucions són benvingudes! Si tens suggeriments o millores, segueix aquests passos:
1. Fes un fork del repositori.
2. Crea una branca nova (`feature/nom-millora`).
3. Fes un commit amb els canvis (`git commit -m "Afegida nova funcionalitat"`).
4. Puja la branca (`git push origin feature/nom-millora`).
5. Fes una pull request.

---

## **Llicència**
Aquest projecte està disponible sota la llicència MIT. Consulta el fitxer [MIT License](LICENSE) per a més informació.

---

Si tens qualsevol dubte o suggeriment, no dubtis a contactar amb nosaltres o obrir un issue al repositori.

