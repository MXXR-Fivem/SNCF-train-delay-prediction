# 📅 Visualisation of expected delay for SNCF trains, EPITECH Tardis project (Data/IA pool)
Visualization of the expected train delay for SNCF using the TARDIS project, highlighting delay patterns and trends through exploratory data analysis and modeling.

## 🧾 License
This code is published for demonstration purposes only.  
© 2026 — **All rights reserved.**

## ⚙️ Tech Stack

(clean of database)
- python
- pandas
- matplotlib

(models benchmark and generation of model.joblib)
- python
- sklearn
- joblib
- pandas
- matplotlib
- numpy

(creation of a dashboard)
- python
- streamlit
- pandas
- matplotlib

## 📬 Contact

Théo Busiris :
- Email pro : [contact@busiristheo.com](contact@busiristheo.com)
- LinkedIn : [linkedin.com/in/theobusiris](https://linkedin.com/in/theobusiris)
- GitHub : [github.com/MXXR-Fivem](https://github.com/MXXR-Fivem)

Raphael Leger : 
- Email pro : [raphael.leger@epitech.eu](raphael.leger@epitech.eu)
- LinkedIn : [linkedin.com/in/raphaelleger](https://www.linkedin.com/in/rapha%C3%ABl-leger-8549a8336/)
- Github : [github.com/zzKew](https://github.com/zzKew)

Zachary Joriot :
- Email pro :  [z.joriot@gmail.com](z.joriot@gmail.com)
- LinkedIn :  [linkedin.com/in/zachary-joriot](https://www.linkedin.com/in/zachary-joriot-66b8852b6)
- GitHub :  [github.com/ZacharyDevProjects](https://github.com/ZacharyDevProjects)

## ▶️ Run Locally

1. Clone the repo and go into the folder : 
```bash
git clone git@github.com:EpitechBachelorPromo2028/B-DAT-200-PAR-2-1-tardis-4.git
cd B-DAT-200-PAR-2-1-tardis-4.git
```

2. set up venv with python :
```bash
python -m venv venv 
source venv/bin/activate
pip install -r requirement.txt
```

3. run the tardis_eda then the tardis_model :
```bash
jupyter nbconvert --to notebook --execute --inplace tardis_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace tardis_model.ipynb
```

4. run the dashboard : 
```bash
streamlit run tardis_dashboard.py
```
