{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "815443a9-c7b0-4e4d-b80d-e7b6c57508e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import lightgbm as lgb\n",
    "from datetime import datetime\n",
    "from sklearn.model_selection import train_test_split\n",
    "from darts.datasets import  SunspotsDataset\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from darts.models import TCNModel\n",
    "from darts.dataprocessing.transformers import Scaler\n",
    "from darts.utils.callbacks import TFMProgressBar\n",
    "from darts import TimeSeries, concatenate\n",
    "import my_get_time_series as mgts\n",
    "import my_metrics as mm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "be8e2fd8-2865-4f8f-80d2-b43980ac5d37",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>Ventas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2017-09-30</td>\n",
       "      <td>5,783</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2017-10-31</td>\n",
       "      <td>20,854</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2017-11-30</td>\n",
       "      <td>14,191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2017-12-31</td>\n",
       "      <td>7,595</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2018-01-31</td>\n",
       "      <td>12,358</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>72</th>\n",
       "      <td>2023-09-30</td>\n",
       "      <td>16,520</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>73</th>\n",
       "      <td>2023-10-31</td>\n",
       "      <td>17,195</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74</th>\n",
       "      <td>2023-11-30</td>\n",
       "      <td>13,819</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75</th>\n",
       "      <td>2023-12-31</td>\n",
       "      <td>21,319</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>76</th>\n",
       "      <td>2024-01-31</td>\n",
       "      <td>3,791</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>77 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         date  Ventas\n",
       "0  2017-09-30   5,783\n",
       "1  2017-10-31  20,854\n",
       "2  2017-11-30  14,191\n",
       "3  2017-12-31   7,595\n",
       "4  2018-01-31  12,358\n",
       "..        ...     ...\n",
       "72 2023-09-30  16,520\n",
       "73 2023-10-31  17,195\n",
       "74 2023-11-30  13,819\n",
       "75 2023-12-31  21,319\n",
       "76 2024-01-31   3,791\n",
       "\n",
       "[77 rows x 2 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ts = mgts.get_ts('ventas')\n",
    "ts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4532f8c3-b0fe-455c-bd91-27955e829ead",
   "metadata": {},
   "outputs": [],
   "source": [
    "ts = TimeSeries.from_dataframe(ts, time_col='date', value_cols=['Ventas'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6b97a538-f212-4347-9ff4-a0f6694dd2da",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = Scaler()\n",
    "ts_scaled = scaler.fit_transform(ts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a2ce9384-50cd-479a-ac3b-636f42190a08",
   "metadata": {},
   "outputs": [],
   "source": [
    "split_train_test_index = int(len(ts_scaled) * 0.8)\n",
    "\n",
    "train_scaled = ts_scaled[:split_train_test_index]\n",
    "test_scaled = ts_scaled[split_train_test_index:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7ed3a76b-db6f-4d30-8900-fa0dd3debb0d",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_name = \"TCN_model\"\n",
    "tcn_model = TCNModel(\n",
    "    input_chunk_length=7,\n",
    "    output_chunk_length=3,\n",
    "    n_epochs=500,\n",
    "    dropout=0.2,\n",
    "    dilation_base=2,\n",
    "    weight_norm=True,\n",
    "    kernel_size=3,\n",
    "    num_filters=6,\n",
    "    random_state=0\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "0ab20b65-a0c5-4fe2-ba2d-7dfe0cf32d90",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\jcoqu\\anaconda3\\envs\\aa2_37\\envs\\TFGinso\\Lib\\site-packages\\torch\\nn\\utils\\weight_norm.py:28: UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.\n",
      "  warnings.warn(\"torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.\")\n",
      "GPU available: False, used: False\n",
      "TPU available: False, using: 0 TPU cores\n",
      "IPU available: False, using: 0 IPUs\n",
      "HPU available: False, using: 0 HPUs\n",
      "\n",
      "  | Name          | Type             | Params\n",
      "---------------------------------------------------\n",
      "0 | criterion     | MSELoss          | 0     \n",
      "1 | train_metrics | MetricCollection | 0     \n",
      "2 | val_metrics   | MetricCollection | 0     \n",
      "3 | res_blocks    | ModuleList       | 309   \n",
      "---------------------------------------------------\n",
      "309       Trainable params\n",
      "0         Non-trainable params\n",
      "309       Total params\n",
      "0.001     Total estimated model params size (MB)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Sanity Checking: |                                                                               | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ebd6ee80b28d43628fc671157b8ff91c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Training: |                                                                                      | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1e9ef2d09be4565afa4bc5a00c0da3b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "38f26be3694a47508afa4d5dc2918736",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d1bf7329225b4cca99d34c8d18e8931a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "571ed86fd491483eba5dd60025802c84",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "69cda06d6d83471ab29d8be339c200bc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7ddc8869da684aba8098e46c365638ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee8cecc2e28444e68229e9afab816da7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7f36e5e5841240648b9c9e105fc0ac95",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0a2f074bc39c4666943635f2fc9dfc13",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "db9ebf489f3740f58b6268970ee0bc5c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4faec00e88a94df89e726890fa46e3f3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6c07dfe661e14c4f9270658d69e3569a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d6403b68ecef4959ab864f853a3bd798",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "728933978dab441291826f246c824db0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "992c99c5c7134df38a99e39ba4308460",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "08aca6a9184e44dda428d4222f48ebf4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fe683dbb90fc4d5c9b6fda26378b74d8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "721f4a3d6a8747f69e6ec71bfd276c31",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0487197352c3464fb7bf8bfa56eb3339",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "26eda92fd7b44543871134e1aa8fa4c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9640de4f938d43829c6894c3f26cc850",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fd42bce42a854262a83e68bf467be63d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8ad4187bcb6a48eb8d068f918804f5e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "712a571fe0034ca28ce955b5fd5b00fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cf22e010fcc14a3f88930bc857afe537",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c285654d18a4710b9dde75dfc4460ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a1f1db202a844d05ae0a9eee26694155",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "23b0c6c4390f42288b1332a88c07a41b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "154339dff0a2468a929e31492ba26473",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d99f2f5459384157a7115d3090d665fe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e0610b024f1a43d6b96110ce05b023a5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "46b697bafc844119a73f719a9eec7429",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9787b387354645278cdafaa3596342da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1ee4b5d7bac44aa09f45c6d34a271315",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "afb04eb3c1d84eb1bea875afe0817df0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15448ab8d8194c04b0b6ecbe035d982c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "30be119b8dd642868b20a4f0a4f6f3f6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "707778314f9647aa912b2096a9c295d6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "10d79ee104f34dab92dcce34c08c7b60",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1b5d68972874d78a05f8f7e086d476a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "567d7bc3e939431a81084e20eb305692",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "da7b0f2980954f619cbd9efe5859172b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9bb4f4c4775544eebf0b27a2513cf76a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b31a6b77272740079437011aa42fa47a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "075cab92b90442f2aa031973d88d83d4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e41e33f999db49f68f6440632f8a960c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e1c99701fe9d4f699cce221575e56e42",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3f20f01de23c4cc2b7b488a614a8505e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ffb141965ae44ba5a6b406c7ea3958e9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "05799fd00f18401eafddab0cb1a263cf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "397f320acaae4f918efbcc11a4d7dc43",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b66590e371f74001ac0e441c22158a99",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a309ede091c14f3fb2a810ef1fd73c2b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3ee67a161a784f1c939476e24f97304e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "63344f83845a4d6787cf2ebe7ae08a14",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "485245b631904a988d2a6f869ae44a2f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "77331fd2448a4e47828077e797fd3272",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "985a36c7e0cd4edca248e4c4dca0353c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6927e8cafb8c4f78b3bec7fdc1588c92",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "021c7f518b9f43ccb6e9302c85327a13",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56ad3b60b12e488ea9a89228cf784ecd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a3db67ee98c347bfaca09dda7dd7d30c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ab54fbf48b9a43c38a94a942cf63b5fa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67abcf7a626d4ab8b7c73ad653f37ac3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a8a1eabf56694f6ea23788f6aaca50ab",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a3d1ab2bcbb48b581a7a9b54c363950",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6cf1801ac1e148c1a7e9926d2447d7dd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "af7ed243b0ed49c38a71f127b5eb1132",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b1bd94406c64d1e9a690a77f1888e5c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ca6386163d1f49dcb9ec14813bb7a304",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cbe3426a07e14bce85c5e8dceb30c626",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "44f7933483e94a64b084a63b898e99d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b6de6edd88bb472883d13964add53d3e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fb5ee4bf75724ab49ca9e2db103d1df5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "89f8d7ae09f144f2be220da348aa1216",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d78777cd5fe54d6f89aca82d58bf5a98",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3e3d7cdd54ac4ba6ac7b7b491aabdf6d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be08975b20f5465b9bdb6331795cbbec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bbce1338be894359906eaba5fa334f5a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8700f6e966614533a3d3a448814ad29b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3edd4c905a824f99857fdd80011d7136",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6734c18b6b7c452a924a21d46d634c20",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d957edf048224e5e82bd1e982725687f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "64b1312b52524c3185b6ffbf180c33b9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "508100f19f36487297d2dd154e56dd7c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d94ece03b1c44eb594d259f7b81a4e47",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4721577fa86b40b9a1e9113a2a3b42a4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "136eede66ad04a2fa59257dfd2372c01",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "05077a2d80c140e7bfad4449b6d331d2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fcec3d5cdf54429e95acac368d122c88",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fb20b28485364c7296b4e649d1f48215",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8af01a26c2564ecfbfbe27b719c9e3a0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "34d2631a772b44cf9448056e2681bde4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4f8faadd6a674c97b5ef5d94289afefe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0d217c542ec444d493cbb882c8da1eeb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "99b4f3481d784676b61750c3d82c4a77",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7200aa13073442ae9c38ec57c5e0851f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2b893c3d869a46ba9db26fb4f9bc2903",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "af283c05792549aab9ee76b934a98a30",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "170d0a8b4f7045c493362bac204cc0fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "522414c6e2ab48fa9f7201213ca0fda0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "564953ed279645fe81d737eed7d0cdaf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ed5c344688dd4ccf9a0f2fe703d84afd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1d4833594941446785174648c1c22189",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "73f34913dfff47c9bffc8143990d0310",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be0a7ee8e701483d8f24d6f18b065075",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a993280dce894e9ca3b272476f48b557",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a38ff12621c547faa4cdcf0200be0862",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aaba4cc7c86b4ce0909d32e444287a7b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bc1e9b7d0d1b4b488038ac544906632f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3a0c2dc57fbc43cdb886328ddc65ddfa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b0832000a6724f2ea0da83c668f6979a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6e7a19386dfe4b30bbc2a1d4fa9ba2d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "42b76fd95400492682026dbd54df90b2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "88eb4e9999a04f4ab7a48959417c5d75",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "782d6973dad04c2a9475c1b3e686cfdd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cf33c257040042bb9bd8b8f1f0aeebac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0f6c941a2ea0432cb2dfdf354935a397",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "448b56a0645a49eb9e03bdcba3b071e0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a9d359ddd53042ca83146139fe02f2e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4e0fd7aa51a14ce7bc45ce2686da1ba3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ad56621ef7494eee95cbe72f3fabf498",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eaba9264398b4ecbaf70efe0a4e79cd5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b8f09bda96be418cb2d59509df1cd416",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "13ef69a5056647cca276b3ecb657a526",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "41c401caefad4abe9e44e67563749d6b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "07e6e434bff74ab29b8833035ab210ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "057e303a307d4fe999c784102a7eb0a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9229ac45c1374d918449f3b233392386",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6447c5b5ff694d0ab30d9003b360fc1c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ef21bed13aeb4a7cbdfd2c4b125815e5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3465494d9c3b4ebc9fb5dcf32620bc3b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "89a91b0e35a7431781bda3ce3f58f958",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56ca73b07f7b43fe95b926469cb6c367",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4770f61e8dd4417381bf66699e3cdcfb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d37f4a72069493a94864a3ddd7107ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "67f4d0dffc6743f881fe9d7cbe8e363c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "55a8acf761cc4fc88cbee721ebf80fe8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8de9c410ac8746de9194f7e687651f62",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "65817a232ca34ac1aa0a66ee9cdfa0d8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "38e717fcd9ca4663a4711466a62e3eac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "638eed0627e140ef98ddc51a7971788f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "28594f0ab9314a599e2bd553d8fec911",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1d9af066108f43709de4ff716fad6913",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b9659a46161c46adb73e1d83d82e5b14",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79b6243b261b4f8589a097cbe5254b19",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "188a1b2838014dd7852f29a84ceb8426",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bd16c7cd183847f7aa68fd5083cfdba0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cbf906f49a094a74b0bcffce8ef85167",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2b05ab23317e432096c887e93595406c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9925708a46fd46e4b4163675fbab80e6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "05bba4b19b744748b35dc23f6836fa28",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b4ac69b2240463590970cd0dd6d63f2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9e5205d8ba894e88b6b55abe9b0522a6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bf3016bad87a4a869395454dbc758759",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "690303600ff14a5c939304d5fd4a1ce5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3acb65bb9fd7468db92b35e22c05749a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cc3e5df4f1774e44a4e70e6a6ea4658e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8d3b2883e06244c18c261ee5e9bfa142",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fa6261e3699a4a9c887f40545a078772",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2d19c2fec2664fae99c33e6e786bb52c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1b79a6932f0f42678dd33def2b662cde",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fb006269825c4a49bb5fc6dc57e848a2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "805174f351ce46769a3ef8b812ec120d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b6c08ccb15b44a5ca149bbb0bda21135",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a6cec5e8e9b14352952236a298d0a028",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "00f0db7670a942efba2e110945a7dd49",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "09263cc5460e4564b34d86d1884d574b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ba7472db3e754798a453b4fe1c05fc67",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3efef1f527124e9bb43ccb4a532cdff5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f70cd971c4e4480f8be06acb6cdeb562",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "338021da2f66455a935f08620c33eb75",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c25c106c410a4a1ab388b96d6e0d32a6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "958d94a0be5e4ad5a24ad5b622bc194b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "de84d015173448b2bf9575cecda917b8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4c4a62577e3c46b08d80dd21297662ad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c4ad9395b55648fca0705660dfb6e5f7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ef65abdba1b24ea9b9ab6f8a24bee967",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b50c0bdabae943d0b7859a77e3c0fd48",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2f1631a2689e4e0c95ef9816b50f7cc8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1781e2579a9c4191be85c36c58ebabe4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f3ab29e7591e436cb1a47af6ded9a4ba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ace5f2c28ed14d56ac91c838c56dd6d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a39642a6a004e0fb43672b7abfc4de7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2414181be08f45ca98815afbf45e5597",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2d06fccfccd64c85bd3ed2be748ff30c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8ffe7d474793486788a7f0507f3b30c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d61b1875d05f46e7b466cacc1ac3a783",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5b432f4d17a8465ba67ba97095584eb7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "10cc96c8a11a448abc583e77348925ae",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1766dc09b30e4ef283399d8188d933f0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e1fcf7283bd84ca49d596f0616b18b9a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "188933c6c4dd432bbb7f76edff01fa53",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "156ea076b345492292707ba6154ac8d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5a98c6c1b9534884958ecc71f10518c6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "107de92e246b4164a34ceb382565c2ef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fff910ea007d4cc5bbbe88c2cbd57a52",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a13e69ae12ea4290a1ab1caf2f71b5f4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "722d127792324ace84716f8007c7951e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bc20182a789942fcad71d6071d6deecd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "814423547618411fa5a23742525976a5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a5c96e3da25480aab9763b59ac4b8da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7ba8eb07d0114f2ea71f5b7c513788b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b32794e0049c4bbe9eb84be81848c3ef",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "207cbc8e2d564ecd8708460d20699a5e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6d564f3388c44c01a542002be8a35058",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fa9e226a26b24b79bb1b9c97561a07da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "76a70a295cef415d85607046b85d084b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "965f5e2d052c4adfb1fdcca6c1ae4511",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b007f4d7d4874f1ab7ff0492dba53717",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a8f0ae73737c423e906991d69fe3d704",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d6bc276d1a614e19831f5ffec9cf281c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9a7fd97b523c46b79611da58d4ac5d34",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "04070532fbef4ef2b910ef8f473d6760",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "113c53bfd3be4a34840deb74bdc23878",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "01b5bea864e046419313f3663c783011",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "75c9f88824234afba35be89b175ecc88",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0523e48231e34682ad1f3cf9270bc43a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b96beeca87074d6c9ef3cc9f548a9684",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5dfb2a354cbf4dd993476a8b94ee3a14",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "215ade0f42c04e43ac18e83b7c83d39e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "72d3352ee8ad400fa1207909466661ca",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a28069365f5c42bd82d1d91cba7bfa46",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4e8b2898b9a34439ab171a88168b0825",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1963f6573b524dd18d908b2d998c1cda",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9689829371fc4b99bb88274b75b46e01",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1f4e68e464a34f7f9df84d92f669029a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a0b551d74a8643a8852ae15bfc4f48e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "07889a85987d41849ccf98c542e8cac7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a8d220507bd4756b7370a53eeff7efc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "32bbd24815c9457c8c72d73dba9ae97e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "28a96dc5f3b04324b9accc8f59f13f41",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "43cb76316b0f4f1ea2551adcda431c55",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8d53b9e855a543739c4fe5dabeb11a74",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2371051a597d427fbb36592b794d81b3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3287178f77f4451fadf398eeb2740d1b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b362d64b0f2d49dcbb5d33aa85d2a8cf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "14ec190e87124e1cb9f40b6a550a2f28",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "13a7f243914b4a38aa1b2e3b6c01d22d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7399f6de5f8e4198b0c92a2495157208",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "765c908132c94d3f8b4b370bd23bfec0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15d2edcec71e41ee897a92059fad4233",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1986921582164228ad45b83609d609e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fc5f14eb84714280afa2f754251c4519",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "12cd5de5612d4790a8490941606a08b2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f4f87297c5e44e7a9db6536178f714dd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4f304a726595434ab69d529dd5aac7a0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9c6f875c56da4b2996384e3ec64a8856",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "31b3abea081042cd996ae4d652c023d4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d5663e784233483eac610ada79f9b8b7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2c2de0e44c5d4b59aca9ad7f88889df0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "abee924f4de245eaaf237b0869efe65a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c7875d6031654c539eca720ed0376fc9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "829b0cd554cf421ebcc3c1b3175a24f3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1e59b86238a84ec48bbaaa533de2825f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f645e6b535204f5e80523e4dad6bbaae",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "45e23fbd28774f99a5f9a817f0f5b145",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "46ece02d31f241e698c5a88ee4959bd0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8a5c42f77d404eb4971b322d624a0e21",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9bc7d36c4c30498d9e4b4b26542e0f00",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a4de9a7da6384a12be4b64285b0b858b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c2dc10b8258e4e5993bc67f2d8631fc2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4dd2a14500c14088a7fcee08f84117c9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d4dde64c831a41a986cc18bc1dda53ca",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e012b5e3c5a54514ae6acca25cd8c37f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "32b268407b834753b71e7ec6fbcfa434",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c3e221361571431eb038091a49804baf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "624371fe2d8d4f37b51cda3dd25dd516",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6274d2c334da429fa46d437a7a9ea007",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2b761ff16b044075a05709eea2b74411",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9bda07312f4a4e689b6c310fa346691e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eb166c2ce24b4b68b830228b563289e8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b02be3ef01c449bb88b06688c3f7cfdd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "53243442a9574a4c97979dcf9340cc5e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e665343712194e2d8f614a541cbf707c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eb5a95789c3a4d8989ce7b8174a11b31",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3092da1fd0ad4b2ba041559f0d0e6d5e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "59d16f01b2ab4cdf9b1be49e547650dc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e73a133727c146fd8ee96eb33851685f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0b1ecd2b140b4c80b3651536fb4a4f44",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1754361622ed464d8765d1a6b32979a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "560d9826b11e4aaaa0587a692a9d347d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "abc34242d46d4b5e9838a9f0efb3cd6f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9e6dc2bc7d4c43d9aede58dd73a3b91c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a164ea0bd59e439986e666880505a3e8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "20890b285b274ffb8215b3320ad1e007",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ece7ca32b7204aa79e6667f2e8d7b8bb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "250468baa2be44b68ff28c0cdcb775a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "77f7c8b09678409babcac8f2bc283673",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3ace36690b014380bf83a137a16ff3f4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "37b0449c643944c984eee42550c4ac4a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9667162aaf9f464e92a34684fe7962a0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8d8c7bd5fe474dffb37808d1dbc274e3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1f3123f805e4602b082d64ee46e87c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee696f5473ef4635965591cdd027830a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dcf9a7f5051744c296061934bb0c0531",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "55345381eb59477aa61741546f189870",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e8f0086954194fc3b065ab8ef309fbec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "249ebdb4f67b49ba887f36e3f17426f7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "45c9c8ab46b34b5990bf45ad95630bff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "85da9432fdef4be7bc390e4e466c8cea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "97d73eb5a5ac466d93ee7376b526338e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "510e18d116994b4eabbfee899d419d9e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3485a8e8a35c4dfd9729201a308bd242",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9fa203b1c280439e993a6efed3479823",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f4ab2a932d4b4c078193540296bd3d42",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5ce26ac677034a5ebf9396b433bd03a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be957fbbc7e741d88b90af0d1367baa4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "509b032f1e3d44b5a0fd06eba157c224",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "89a270eb6d3048ccaa7b9f6cf0f578dc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "953c1e46b0f4450fbc5e441f6a0743b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "19af333e17604e09b33eba47657459fc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9492b0344efb4909b8f1df6cfb2f0dff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15ea677e1b344f7f8faafda920332395",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e77ddd7012d54dbabd2bc412d13c527a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "45a600e0ffb2429293b20784f09a80e9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c53d96d3e6540e78fa8b3df38639ffa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fffc54d9205d43f5a5f9d42b15280bb9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35fd2dda8e0b4b239690188c4cb0613e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b27a67156ee40fbbd3bd74f145ba7f9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b7eb79618dab48e4958e1fd3c5d66cbc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c4ade59326f4bab8dc7bc2ee45738d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "adf952b74e45477a945e5b7262d9b2bb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d79218e4acff4c8f9da2b5b2ca518b3e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d07c7ec2d9274356b3cc46ebb54a81ce",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35f9e112c5d24119a6cec0edf6d2eac1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4c91f45f761946eaa7f5999f31a43055",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9b1fd6c09f3447daa9dbd53f843f5837",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "05a047c752da4039b6ba5211ef126dd1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a59d3394aa7d441c99d618442295ecb6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e400e80ed9d84a3abf223b95e97e3a5a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ae71e72614a84f248aa8b6d572eecb37",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "029295048e74487e9ff20a6dc8c53be8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "af4bbd10df484aa88c3e71d82e8c3e7e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "87ee0dc6f84a4a95a5718e143b68866a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cf1bc3dd6076451a88a4314d74bc68c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7998424cd17e459396e43a62098fc3cc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c04072c791934b0c94f95d9cfa155073",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a9d9d4789b8476fb272c2b5ad3ccfed",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3aa2b2b6c897484aa465fcbe79e0dd0f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9311776c73294ee7ae1de83fc74c8bd4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d99314b9147f46c48a818f946c910a97",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e717cb85e7a84f8e804063e8827aa05f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5d60f7c570454ea68c41a7887653b173",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6c2261f0d5ac425eb0d8673a78c60963",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1074fed4a85545239cc0778fa6bd0c9e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d9ae68e02dd49b88bd0b1124e018bad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b53042e35e13433b9129a2ef73253bee",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "62a9199a22824544bde4331b11a5e6b8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2323f35f0b93490c919dd808e36f9413",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6faa03cfdfea40efbe803b3c35ea9d63",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5fe44ed1b41e49cc9e3774d40468b821",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "74fb9fa993e1417eada08269b3aa8582",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "06d3c439188c4e8a982b8d64e2aa3210",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7bd2fb3f7caf4993a5e8fc94b7981af1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ffde5325fe7244f09ce36e04ad9ae906",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "112bcafc78284814b9c080955511bc26",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9ac796eac8d14a74b117911f1a337ad0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "61bff4b99a054d409dffc2b5115fe45f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6197c91bb0a04d3ab4933844b77f3a9a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c1fc7d11fc4c4d14b7cc641138f34b72",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e5f948c751594a9aa01b18467227257a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7c61773edde54998ae9184329196c6d3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a263977254145aebe5a8c2ced40078c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9f40594e0bc64383b903c1ca584f6630",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "90dd0e3dca4c4a228092c6006bf4ca50",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9d17770e0d43455eba50879b3aa13e6c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "717a25ac613148499e5be07b777054f4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e711a42553ed47a8b8c456da5fbd0043",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6eccb69eb22d49848ab26a377cbfe998",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1d999c93091340cdb279eaf863cd8625",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56669c02f3df41c6bcc747f3922205d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c544daaef3f64b6fa6a0f940d6a54eba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "309c81bce2bd443d903ca51c33520577",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f22e873050aa4d0e966e00212fc02017",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3bf3fbc8de924fe9aeadda61c14678e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6399e20c5a5a4c988709146d3bdca2af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c8bb00f128e449c78d4d518225888a83",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4cbaf574816445f88aa3d228aa8c9040",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d72eddd8be43430a82c52ffb248823d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "533006ca51e04a4d9a4e9a78989a20af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b984be0b3be04e889d34b14d29117c3e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4533de7bdfee48a9980437a9a4397d3f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b2e08e6373249058b513a21bd36fa40",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35841281a9b1473db3c720fbadcf3a16",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3506a03bbca84c96af48eb89de780f9b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6b9a4164efb14ae49cd163cf60f0ae8b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2e3a1e3aa5734905a2c7c76f7cce53d8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c39ddb3e03214ede9c817aaa36e98832",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2efcf348553d495b9eafa685fd8a4875",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "21813857621549918336ffdb3b3f3f2a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "50fa3ddb5c1e4ce58acb0458f43c95be",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9a755feb42fe4416bf652aa75f298a4e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1caef34782124ff7afdfb578af6924b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d4961920ee7d4ac3bbb7c1cea9f86a0d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "004e2d77d5b94ba582484c25a5a4d44f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "217bedbd82e249a381b80b2e0ed24a1c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e750f80bb49e4d8d96f1f79dbc650b93",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "58ac5554739349e5879956a069b768d0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8beca3ddc28d45888fb4e3765e4f2fb7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "deda3b6f13e946568ccd0975f608b865",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c4a217e938d443fa2acc6a675fa30c2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "50913902ed3146299e4736aff28be95e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9adc10bdb20d43c7b4d6f276088411b4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a12c42248ab5425bb4bb19f1f1ac9964",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4c35fe5fc54c4fa7aa77f55e0346f79a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d15bdf189e534ec49390df37300625c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5513fcc2ee67490c9e5867a79c6c64ca",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4923ca31a9024e9381193405bda1e72f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d731a66905e244c89e978fce74dc0f33",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b41d376f8e0469c8498a868c9bb657d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "64d6a5e119d648909cace10a7377e638",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d69b862886764c4cba74f34ff469ac36",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "60657fe2919543e9810d9759d507b2d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6af268f6dfa34b019e41fd1000882a29",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "84a79bd6910e4bfa9b8fb3af23265214",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aaaeba66da4d4ac3afd1df8b1e35cbc9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "de89864842eb4d2c8ee397a09ed2b8db",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "625f754e408945469d49f92aaf0f83f3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56945ae151784db09f8bd2e69e127e47",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6f3b6f2ecdb14e2e825b4efd7c05cde5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "788cfba9f2954706b6bddc2fbbf338e6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "692e4fd73ac84f4c96dd274298505b7b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5d76447cad6541199eefaef4f631cd43",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5ef16f17eb0e44f08c22fdb50d278310",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1d39d348f7ea49c1ac66863b51f5ca7e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a65d8399d5fa4313858a6d9a74443e56",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "65d3243ae5524b8da0e15b1745c2fc1f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c6122d0d240453285fd0c12637759eb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0de967c626df4c8b9ff0d28b4e5add36",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "20da876468fc460ba1889095e55f91e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "55759e25f794441da928ea3d0e7a03fe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c135569da68b459790017763b84621e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ead1e50112b04b8592fc7254467082ec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d1f040da5124c08a0d9191a520c93fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6455724715bd4a92be8c6eb21fb66221",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b0fc46b28b614820b49b0f3a4f142121",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "81dcd011f5ce43f4b8313577b398696a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "756238d74a0741eaa65bb8bd09380179",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ad97d75abe6340c3ab2aabc962d45cb6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a62a9faf7afb4c31869f4b73992e3751",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ea8757460ffa48c2b2036598d4bbc189",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b09a2b2039364e128922a74fdefd1d16",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bc04cff7f9da4386905551aa602d15e2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a1d9f623e03a42578aa4692be9c4f49f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "43d560e4e18f4d1eb3f60f346f76513b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c2d05aa51f284b26ab255d17dd986f94",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4413a0ae26e34f20b66a5a13692a9ace",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a790d5e7cb344bca00a7b434e705f4e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7000fd17e99d4c38ba4a2ea12d302cf7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5065be2da61d4bbfbce09d1ea60e1aa1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "102b7fffc12a45879d8490be33289789",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a1591592dc53451daaa1aaee104d7396",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Validation: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "`Trainer.fit` stopped: `max_epochs=500` reached.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "TCNModel(output_chunk_shift=0, kernel_size=3, num_filters=6, num_layers=None, dilation_base=2, weight_norm=True, dropout=0.2, input_chunk_length=7, output_chunk_length=3, n_epochs=500, random_state=0)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tcn_model.fit(train_scaled, val_series = test_scaled, verbose = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e1b0efbc-05ae-4398-8359-10bbe9732231",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "GPU available: False, used: False\n",
      "TPU available: False, using: 0 TPU cores\n",
      "IPU available: False, using: 0 IPUs\n",
      "HPU available: False, using: 0 HPUs\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b7192c1cc5244003ae55f75f51c6c662",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Predicting: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='date'>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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40HaS98wqm+OHj0qVldXi6IoikeOHBGvu+468ddffw3GLvqMUX0j8cgjj4gTJ04Un3322YDsDwmZADB06FBx79694pNPPinOnTuXPf/NN9+I9957r8v7b7rpJpeL9Z///Gdx8eLF7O8333xTfOWVV4xrdIDQo28GDRokHjx4kP09b9488a9//atxjQ4Q3vaNKDoG1n79+rG/r7vuOnHbtm3s72eeeUb85z//aVibA4UefSOxYMGCkBUySujZNxITJ04Uv//+e93bGmj07pujR4+K1113nfjjjz8a0t5AolffbNy4UXz44YfFf/7znwETMhRaMphjx47h4sWLyMnJQVFREdq3b89eKygoQGFhoabvuemmm7B27VpUVVXh7Nmz2LhxI3r37m1UswOCXn0DQBaGEkXRq8+aEV/7Ztu2bcjPzwfgCLmVlZWhoKBA02dDBT36Jlwxom8uXryIwsLCkO87Pftm4cKF6Nu3L2666SY0b948Ysdi575paGjA7NmzMXXqVMPbzENCxkCk/IQJEyYgOTkZ1dXVSEpKYq8nJSWhpqZG03ddcsklqKqqwqBBgzBs2DB06dIF/fr1M6rphqNn31x99dWYP38+qqurcezYMXzzzTeora01qumG42vfFBcX4+2338YDDzwAAKiurmbv9/TZUEGvvglHjOgbu92OGTNmYNCgQcjLyzO0/Uaid99MmDAB69atw8KFCzFo0CBER0cbvg9GoWffLFq0CH369EF2dnZA2i5BQsYgrFYrnnjiCeTk5ODee+8FACQmJqKqqoq9p6qqCgkJCZq+78knn0THjh2xdu1arFy5EseOHQvJpFZA/7559NFHER0djTFjxuDRRx/FsGHD0Lx5c0PabjS+9s2ZM2fw4IMP4o9//CN+97vfsc9J73f32VBBz74JN4zqmxdffBGVlZV48sknjd0BAzGqbwRBQJcuXXDmzBl89dVXxu6EQejZN6dPn8Y333yDe+65J3A78BskZAzAbrdj2rRpEAQB06dPhyAIAIC8vDwcOnSIva+wsBDt2rXT9J0HDx7EmDFjEBcXh4yMDAwZMgQ///yzIe03EiP6pkmTJpg5cyZWrlyJzz77DKIoonPnzoa030h87Zvy8nLcf//9GDNmDG6++Wb2fGpqKjIyMnzuVzOhd9+EE0b1zezZs7F//368/vrriI2NNX5HDCAQ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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "scaled_predictions = tcn_model.predict(n= len(test_scaled), series = train_scaled)\n",
    "tcn_predictions = scaler.inverse_transform(scaled_predictions)\n",
    "ts.plot(label = 'actual')\n",
    "tcn_predictions.plot(label = 'forecast')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "af69e8da-ac1d-4965-a9fe-d518552cd796",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='date'>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Wq1SqVIlBVDKlSunZGVlKZGRkUpkZKTi5OSkAIqrq6ty5coVcxfVZIrrdc7MzFR8fHzUv+cnn3zSpNe7n4LE+/TTT6vlvXz5cjGUznSkj4zI04wZM9Tp6l9++eU8FxF84403WLp0qTpfwbJly3juuefIyMgo1rI+zKpVq/jkk08A3ZDi1atXU7lyZfMWysZVrlyZ6dOnc+nSJWbMmEGlSpXUfRs3bqRly5YMGzZM3VbYodfi/jQajdpP5tatW5w+fRqAqVOnkpmpG0P7xhtvGAzFFYXj6OhoMH1DixYtzFia/Ll7CLY1k0RG3OPixYt88cUXALi4uDBx4sT7Hjto0CB++ukndVK0tWvXEhISos4NYm4nT55k6NCh6v0FCxZY7NwOtsjd3Z333nuPixcvsmDBAry91clIuH79vwlypKOvaQQFBam39+3bR2RkJN988w2ge23GjBljrqLZnKFDh9K/f39atmypTjBoyapWrarelkRG2JyJEyeq39jefPNNqlWr9sDjn332WX799VdcXV0B2Lp1K0899RQ3b940eVkf5Pbt2zz77LOkpqYCuqRrxIgRZi1TSeXq6srIkSO5cOECYWFhBjV8VapUwc/Pz4yls125J8bbv38/n3zyCTk5uumhx44d+8DlIETBlCpViu+++469e/dSo0YNcxfnoXLXyFy5csWMJSk6SWSEgRMnTvDdd98BuiF6+Z3ttlOnTmzdupVy5coBcODAAdq2bUt8fLypivpAWq2Wl156ifPnzwO6posvvviiWBfvE/dycnJi8ODBnD59mh9//JHBgwfz/fffy+tiIo8//jjOzs4A/Prrr/z666+ArunPGmoNhOlI05KwWePHj0f5d/mtCRMmqIlJfrRo0YKdO3eq/U9OnDhBq1atiI6ONkVRH2jWrFn88ssvgC4hW7NmTaGm0BemYW9vT58+fQgLC1NXuxbG5+TkxBNPPAGgzhkDuvd26dKlzVQqYQmkaUnYpD179qgLxnl7e/Paa68V+ByPPvooe/bsUatWz58/T1BQEGfPnjVqWR9ky5YtvP/++4Cuw+PKlSul6UKUWLmbl0D33pYmViFNS8LmKIqirvwMMGXKFLVKuqBq167N3r171ZlCY2Njad26tcEsrqZy8eJFnn/+ebVW6YMPPqBz584mv64QluruRGbixImUKlXKTKURlqJKlSrq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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "test = scaler.inverse_transform(test_scaled)\n",
    "test.plot(label = 'actual')\n",
    "tcn_predictions.plot(label = 'forecast')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9332db29-0f0c-4f24-8e6c-36d31e239626",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAE: 3333.2139855794508\n",
      "MAPE: 0.33875639037604455\n",
      "RMSE: 4481.795573059703\n",
      "MSE: 20086491.55869755\n"
     ]
    }
   ],
   "source": [
    "mae, mse, rmse, mape = mm.get_metrics(test, tcn_predictions, tcn = True) \n",
    "\n",
    "print(f\"MAE: {mae}\")\n",
    "print(f\"MAPE: {mape}\")\n",
    "print(f\"RMSE: {rmse}\")\n",
    "print(f\"MSE: {mse}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d4ab779b-0b65-42f6-ac0e-84ece36e0e7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "GPU available: False, used: False\n",
      "TPU available: False, using: 0 TPU cores\n",
      "IPU available: False, using: 0 IPUs\n",
      "HPU available: False, using: 0 HPUs\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "92bc86a009a34304ae9c19f521196431",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Predicting: |                                                                                    | 0/? [00:00<…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='date'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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XnlFXLx4sX47rQFatHvo0KHi77//zv5etmyZ+NRTT+m30xrgabtF0XYCHzhwIPv7+uuvF3/55Rf299NPPy3++9//1m2ftUKLtkusWLEiIEVGCS3bLTFjxgzxq6++0nxftUTrdhcWForXX3+9+O233+qyv1qhVbu///57ce7cueK///3vgBMZJbRot94iQ6klHThx4gQqKiqQkZGBgoICdO7cma3LyclBXl6eqve56aab8N1336G6uhrnz5/H999/j759++q12z6jVbsByNJQoih69Fqj8bbdv/zyC7KzswHY0mmlpaXIyclR9dpAQYu2ByN6tLuiogJ5eXkB/blo2e6VK1diwIABuOmmm9CqVauQPLfZt7uhoQFLlizBnDlzdN9nLdDy+/7ggw8wbNgw3HnnndizZ4+m+0kiozFSncP06dORmJgIs9mMhIQEtj4hIQE1NTWq3qtLly6orq7G0KFDMXLkSPTs2RMDBw7Ua9d9Qst29+vXD8uXL4fZbMaJEyfw2Wefoba2Vq9d9wlv211UVIR//vOfuP/++wEAZrOZPd/dawMFrdoebOjRbqvVimeeeQZDhw5FVlaWrvvvLVq3e/r06di2bRtWrlyJoUOHIjIyUvc2eIOW7V61ahX69++P9u3bG7LvvqBlu6dMmYJPPvkEmzdvxqRJkzB37lycPn1as30lkdGQxsZGPP7448jIyMA999wDAIiPj0d1dTV7TnV1NeLi4lS93xNPPIFu3brhu+++w+eff44TJ04EXOEroH27H374YURGRmLChAl4+OGHMXLkSLRq1UqXffcFb9tdUlKCBx54AH/6059w1VVXsddJz3f12kBBy7YHE3q1e+HChaiqqsITTzyhbwO8RK92C4KAnj17oqSkBJ988om+jfACLdt97tw5fPbZZ7jrrruMa4CXaP19d+3aFc2bN0dUVBRGjRqFXr164ccff9Rsf0lkNMJqtWLevHkQBAHz58+HIAgAgKysLBw7dow9Ly8vD506dVL1nr///jsmTJiAmJgYpKWlYfjw4di5c6cu++8terQ7KSkJCxYswOeff46PPvoIoiiiR48euuy/t3jb7vLyctx3332YMGECbr75ZvZ48+bNkZaW5vVnZiRatz1Y0KvdS5YswW+//YaXXnoJ0dHR+jfEQ4z4vi0WC4qKivRpgJdo3e5Dhw7h7NmzmDBhAkaMGIH33nsPmzdvxn333Wdco1RgxPctCIKsfMBXSGQ04rnnnkNpaSkWLlwoC5GOHj0a69atQ3FxMUpLS7Fq1SqMHj2arW9oaEBdXR1EUURjYyNbBoDu3bvjs88+Q2NjI8rLy/H111/LaigCAT3aXVRUhIsXL6KxsRHbt2/Hxo0bcdtttxneNld40+6qqio88MADGDBgAKZPn+7wnqNHj8by5ctRXV2NAwcOYOvWrRgxYoRRTVKNHm23WCyoq6tDY2OjbDmQ0KPdy5Ytw/bt2/Hqq6/KwvaBhB7t/uSTT1BZWQmr1Yrdu3dj8+bNAReh07rd/fr1w6effopVq1Zh1apVuPnmm3HttdfiueeeM7JZbtHj+/76669RU1ODxsZGfPHFF9i7dy/69Omj2T4LopZaFKacPn0a48aNQ0xMDEymJjd89dVXccUVV2DFihV47733YLVaMX78eMyePZtZ7syZM/Hzzz/L3u+zzz5Du3btUFRUhIULF+LQoUOIjIxE//798dhjjwVMukGvdm/evBkvv/wyqqurkZOTg4cffhg9e/Y0tG2u8Lbd//3vfzF//nyH708aO6S2thYLFizA1q1b0bx5czz44IMYOXKkoW1zh15t37BhA5555hnZunvuuQezZs3Sv1Eq0Kvdubm5iIqKkl0w/vrXv2LUqFHGNMwNerV77ty5+PXXX9HQ0IA2bdpgypQpuOmmmwxtmyv0ajfPm2++iXPnzmHevHm6t0cterX7rrvuYtGcjh074v777yeRIQiCIAiCACi1RBAEQRBEEEMiQxAEQRBE0EIiQxAEQRBE0EIiQxAEQRBE0EIiQxAEQRBE0EIiQxAEQRBE0EIiQxAEQRBE0EIiQxAEQRBE0EIiQxBEwDFz5kzk5uZi5syZ/t4VgiACHBIZgiBCgt27dyM3Nxe5ubk4deqUv3eHIAiDIJEhCIIgCCJoiXT/FIIgCP2oqKjAc889h23btiE5ORkzZsxweM5rr72Gbdu24dy5c6ipqUFKSgr69u2LBx98EC1atMCbb76JpUuXsuffcMMNAICxY8di/vz5sFqt+PDDD/HJJ5+guLgYMTEx6NOnD2bPno309HTD2koQhPaQyBAE4VeeffZZfPPNNwCA2NhYLFmyxOE5P/zwA86dO4fWrVvDYrGgsLAQGzduREFBAf7zn/+gdevWyMrKQkFBAQCgS5cuiI6ORvv27QEAixYtwpo1awAA2dnZKC0txddff429e/figw8+QGpqqkGtJQhCa0hkCILwG8XFxUxi7rjjDjz44IM4fvw4Jk+eLHve3/72N2RnZ8NksmXD169fjwULFuDQoUMoLi7G+PHj0b59e/zpT38CALzwwgto164dAODkyZNYu3YtAGD+/PkYO3YszGYzJk2ahLNnz+LDDz/Evffea1STCYLQGBIZgiD8Rl5eHlseOnQoAKBjx47o3LkzfvvtN7buyJEjmD9/PgoLC1FTUyN7j5KSEhZ5UeLw4cMQRRGATWTmz58vW79//35fm0EQhB8hkSEIIqDZu3cv5s+fD1EUkZSUhKysLNTU1LA0ksViUf1eUsqJp23btpruL0EQxkIiQxCE38jOzmbL3377LXr06IHCwkIcPXqUPX7gwAEWUfnwww/RokULrFy5Eq+//rrsvWJjY9kyH7Xp2rUrBEGAKIoYN24cbr31VgCAKIrYu3cvEhMTdWkbQRDGQCJDEITfyMjIwJAhQ/Dtt99ixYoV+Oabb3D27FlERESwSEtOTg57/uTJk5GSkoKysjKH92rfvj0iIyPR2NiI++67D23btsW0adMwfPhwjB8/Hp988glefPFFrF69GnFxcTh9+jSqq6vx9NNPo3Pnzoa1mSAIbaFxZAiC8Cvz5s3D0KFDERMTg6qqKsyaNQs9e/Zk66+++mo8+OCDaNmyJerq6tCxY0c8/vjjDu+TnJyMhx9+GK1bt8aFCxdw4MABlJaWAgCeeOIJzJ07Fzk5OSgpKcHp06fRrl07TJ06Fb179zasrQRBaI8gSjFbgiAIgiCIIIMiMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC0kMgRBEARBBC3/Hy3LtXSdjvymAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# ts_scaled[:-1] is needed so the time predicitons match with the other algorithms\n",
    "# otherwise, tcn will predict a month after in both start and finish\n",
    "predictions = tcn_model.predict(n= 12, series = ts_scaled[:-1])\n",
    "future_predictions = scaler.inverse_transform(predictions)\n",
    "ts.plot(label = 'actual')\n",
    "future_predictions.plot(label = 'forecast')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "68117f82-f0b7-41e6-a7ec-ab76f68fadf0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>date</th>\n",
       "      <th>Ventas</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2024-01-31</td>\n",
       "      <td>14,980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2024-02-29</td>\n",
       "      <td>15,325</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2024-03-31</td>\n",
       "      <td>14,620</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2024-04-30</td>\n",
       "      <td>14,826</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2024-05-31</td>\n",
       "      <td>15,509</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2024-06-30</td>\n",
       "      <td>13,859</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2024-07-31</td>\n",
       "      <td>14,926</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2024-08-31</td>\n",
       "      <td>14,644</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2024-09-30</td>\n",
       "      <td>14,073</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2024-10-31</td>\n",
       "      <td>14,567</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>2024-11-30</td>\n",
       "      <td>14,436</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>2024-12-31</td>\n",
       "      <td>14,266</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         date  Ventas\n",
       "0  2024-01-31  14,980\n",
       "1  2024-02-29  15,325\n",
       "2  2024-03-31  14,620\n",
       "3  2024-04-30  14,826\n",
       "4  2024-05-31  15,509\n",
       "5  2024-06-30  13,859\n",
       "6  2024-07-31  14,926\n",
       "7  2024-08-31  14,644\n",
       "8  2024-09-30  14,073\n",
       "9  2024-10-31  14,567\n",
       "10 2024-11-30  14,436\n",
       "11 2024-12-31  14,266"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_future = future_predictions.pd_dataframe()\n",
    "df_future.columns.name = None\n",
    "df_future.reset_index(inplace = True)\n",
    "df_future"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "727ffbdd-e2d7-4e4e-bb7a-2a28b9ec07ce",
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
