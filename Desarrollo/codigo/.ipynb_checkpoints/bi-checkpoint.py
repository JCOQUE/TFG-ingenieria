{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25038cad-31d7-43b9-84a1-d90d2bb55a57",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "29cb11a4-827e-4cf0-b511-460a241bf512",
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.options.display.float_format = format_with_commas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cda08066-fb9c-4bc4-afb2-cc1f1729b264",
   "metadata": {},
   "outputs": [],
   "source": [
    "def format_with_commas(x):\n",
    "    return '{:,.0f}'.format(x)\n",
    "\n",
    "\n",
    "def add_grupo(asiento):\n",
    "    grupo = asiento['NoCuenta'][:1]\n",
    "    \n",
    "    return grupos_plan_general_contable[grupo]\n",
    "\n",
    "\n",
    "def create_purchases(asiento):\n",
    "    if asiento['NoCuenta'].startswith('6'):\n",
    "        return asiento['Debe'] - asiento['Haber']\n",
    "    else:\n",
    "        return 0\n",
    "    \n",
    "    \n",
    "def create_sales(asiento):\n",
    "    if asiento['NoCuenta'].startswith('7'):\n",
    "        return asiento['Haber'] - asiento['Debe']\n",
    "    else:\n",
    "        return 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c871c160-c59c-45a0-95bc-73e8dfe383fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('../datasets/diario.csv')\n",
    "dataset.head(6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0214f498-25a2-41bd-9f31-8c081940e203",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.drop(columns = ['Movimiento'], inplace = True)\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "463ed115-c761-492e-9d6a-26ae37571337",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.reset_index(inplace = True)\n",
    "dataset.rename(columns = {'index': 'ID'}, inplace = True)\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "19bf1604-705d-4e32-8fa1-54699ee711ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset['NoCuenta'] = dataset['NoCuenta'].astype(str).str[:3]\n",
    "dataset['Compras'] = dataset.apply(create_purchases, axis = 1)\n",
    "dataset['Ventas'] = dataset.apply(create_sales, axis = 1)\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7601715f-a847-4c46-bff3-795b2ba1a040",
   "metadata": {},
   "outputs": [],
   "source": [
    "grupos_plan_general_contable = {'1': '100',\n",
    "                               '2':'200',\n",
    "                               '3': '300',\n",
    "                               '4':'400',\n",
    "                               '5': '500',\n",
    "                               '6':'600',\n",
    "                               '7': '700',\n",
    "                               '8':'800',\n",
    "                               '9': '900'}\n",
    "dataset['NoGrupo'] = dataset.apply(add_grupo, axis =1)\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "285534fb-57d8-44e0-bb45-ab034aafe319",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset['Fecha'] = pd.to_datetime(dataset['Fecha']).dt.strftime('%d/%m/%Y')\n",
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf249f2d-6ccf-4f06-8604-64e9c04a5ef8",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.to_parquet('../datasets/diario.parquet')\n",
    "dataset.to_csv('../datasets/chheck.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd28541c-44e0-49d6-8124-6503c226aecd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
