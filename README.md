# unAI

Monitor per mesurar consum energètic <br>
`nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,power.limit --format=csv -l 5 >> gpu_metrics.csv`

## PASSES A SEGUIR

Instal·lar torch, torchvision amb CUDA i altres llibreries si no hi són:<br>
`pip install kornia torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

### PYTHON

Hi ha dues opcions: 

Sa primera és utilitzar GPU per tot, se descarrega es dataset en local i se transforma tot en temps d'execució per GPU (`TGPU.py`),<br>
tot i que ha donat amb ses matrius de confusió uns resultats notablement més dolents (executau-lo des de sa root perquè sinó torna a<br>
descarregar `CIFAR-10` a sa carpeta anidada amb `TGPU.py`).
<br><br>

Sa segona opció és utilitzant directament es datasets ja transformats (`TPREPRO.py`), sa càrrega serà pràcticament exclusiva de GPU<br>
perquè ja no gasta recursos en transformar cada pic (en aquest cas es resultats de ses matrius sí que van sobre sa mateixa línia que originalment).

`prepro.py` s'encarrega de carregar es dataset CIFAR-10 (si no, existeix el descarrega a sa carpeta root i després el carrega) i preprocessar-lo<br>
amb ses transformacions per fer diferents tests (`normal`, `horizontal_flip`, `random_rotation`, `gaussian_blur` i `color_jitter`).

En acabar d'executar-se ja se pot esborrar sa carpeta `CIFAR-10` (340MB) i conservar només sa carpeta `data` (586MB) i passar a executar `TPREPRO.py`.
<br><br>

__PENSAU EN ADAPTAR ES PATH ABSOLUT A `PATH_MODELO`__

__EXECUTAU-HO TOT SEMPRE DES DE SA ROOT DES PROJECTE PER NO DESCARREGAR DOS PICS TOT__


### Cpp