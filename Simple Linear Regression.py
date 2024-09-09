{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "69035818-0c59-4845-bb49-72aa1f35224b",
   "metadata": {},
   "source": [
    "# Simple Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f3099e62-694b-4eb3-bf33-3cce0a586f56",
   "metadata": {},
   "outputs": [],
   "source": [
    "# pip install scikit-learn\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "from matplotlib import pyplot as plt\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bb54a90-bab6-42e0-aa87-4ab9eeb1eaf0",
   "metadata": {},
   "source": [
    "### Data Acquisition\n",
    "<hr/>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb13f4f6-5be5-4b30-8aaa-349b2efefe8d",
   "metadata": {},
   "source": [
    "Here we want to predict the mouse weight using mouse height\n",
    "<ol>\n",
    "    <li><b>mouse height in cm (denoted as X)</b></li>\n",
    "    <li><b>mouse weight in g (denoted as y)</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d3f99b2c-c716-4a1c-9913-4d1e67523170",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([8.5, 9.1, 8.8, 9.5, 9.2])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = np.array([8.5, 9.1, 8.8, 9.5, 9.2])\n",
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e507e87a-bda8-48d2-b954-60b896530c30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([20, 25, 22, 33, 27])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.array([20, 25, 22, 33, 27])\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "adbe7e43-d54b-49ab-ab8f-2d03e2162fa0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'y values')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Let us plot our data in a scatteer plot to see whether there is any \n",
    "# linear relationship between the dependent and independent variables\n",
    "plt.figure(figsize=(4,3))\n",
    "plt.scatter(X, y, color='red')\n",
    "plt.xlabel('X values')\n",
    "plt.ylabel('y values')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2009c6a9-d6fe-4b89-b94c-4139ccb265dc",
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       X   y\n",
       "0 8.5000  20\n",
       "1 9.1000  25\n",
       "2 8.8000  22\n",
       "3 9.5000  33\n",
       "4 9.2000  27"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.set_option('display.float_format', lambda x: f'{x:.4f}')\n",
    "myDataFrame = pd.DataFrame(X, columns=['X'])\n",
    "myDataFrame['y'] = y\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ffcf49e3-0972-453a-ad8d-9ab091ba808c",
   "metadata": {},
   "source": [
    "### Plotting a best fitted regression line\n",
    "<hr/>\n",
    "To plot a line the formula used is <b>y = mx + c</b> where \n",
    "<ol>\n",
    "    <li>m is slope</li>\n",
    "    <li>c is intercept</li>\n",
    "</ol>\n",
    "<b>Explanation:</b>\n",
    "<ol>\n",
    "    <li>y is the output (dependent variable). y is the variable (mouse weight) we want to predict.</li>\n",
    "    <li>X is the input (independent variable). We will predict y based on X.</li>\n",
    "    <li>The slope m is important because it tell us how much <u>y changes for each unit increase in X</u>.</li>\n",
    "    <li>The intercept c is important because it allows the regression line to shift up or down to better fit the data. <u>Without the intercept</u>, the regression line would be <u>forced to pass through the origin</u>, which may not reflect the true relationship between the variables.</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64886af0-37c4-481f-9799-be26ecf77247",
   "metadata": {},
   "source": [
    "In simple linear regression<br>\n",
    "<ol>\n",
    "    <li><b>m = covariance(X, y) / Variance(X)</b><br/></li>\n",
    "    <li><b>c = covariance(X, y) / Variance(X)</b></li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3323fc0-7d12-496e-98b2-8aced7665bfd",
   "metadata": {},
   "source": [
    "### Covariance\n",
    "\n",
    "Covariance is a measure of how the two variables, X and y change together. Covariance indicates the direction of the linear relationship between the two variables (positively related or negatively related). The formula for the covariance between X and y is:<br/>\n",
    "<b>Cov(X, y) = Mean((X - X_mean) * (y - y_mean))</b>\n",
    "<ol>\n",
    "    <li>(X - X_mean) -  To find how much X deviates from the Mean of X</li>\n",
    "    <li>(y - y_mean) -  To find how much y deviates from the Mean of y</li>\n",
    "</ol>\n",
    "<b>Multiplying the Deviations:</b><br/>\n",
    "By multiplying (X - X_mean) * (y - y_mean), we check whether X and Y move together. \n",
    "<ol>\n",
    "    <li>If both deviations are either positive or both negative, their product is positive, indicating that X and y tend to increase or decrease together.</li>\n",
    "    <li>If one deviation is positive and the other is negative, their product is negative, indicating that X and y tend to move in opposite directions.</li>\n",
    "</ol>\n",
    "<b>Interpretation of Covariance:</b>\n",
    "<ol>\n",
    "    <li>Positive Covariance: If Cov(X,Y) > 0, then X and y tend to increase or decrease together (i.e., they have a positive linear relationship).</li>\n",
    "    <li>Negative Covariance: If Cov(X,Y) < 0, then X and y tend to move in opposite directions (i.e., they have a negative linear relationship).</li>\n",
    "    <li>Zero Covariance: If Cov(X,Y) = 0, then there is no linear relationship between.</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "31ffdf0d-32a6-46c3-93b5-6475c465ac9b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean value of mouse height (cm): 9.020000000000001\n",
      "Mean value of mouse weight (g): 25.4\n"
     ]
    }
   ],
   "source": [
    "# Let us calucate the X mean and y mean values\n",
    "mean_X_value = myDataFrame['X'].mean()\n",
    "mean_y_value = myDataFrame['y'].mean()\n",
    "print(\"Mean value of mouse height (cm):\", mean_X_value)\n",
    "print(\"Mean value of mouse weight (g):\", mean_y_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d6b5d738-13d0-4b90-926c-5a85f6a8ca95",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "      <th>X - X_mean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20</td>\n",
       "      <td>-0.5200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25</td>\n",
       "      <td>0.0800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22</td>\n",
       "      <td>-0.2200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33</td>\n",
       "      <td>0.4800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27</td>\n",
       "      <td>0.1800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       X   y  X - X_mean\n",
       "0 8.5000  20     -0.5200\n",
       "1 9.1000  25      0.0800\n",
       "2 8.8000  22     -0.2200\n",
       "3 9.5000  33      0.4800\n",
       "4 9.2000  27      0.1800"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let us take every mouse height and see how much it is deviated from the average\n",
    "# Calculate X - X_mean for every individual item in X\n",
    "myDataFrame['X - X_mean'] = myDataFrame['X'] - mean_X_value\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e2976b82-d102-4ef8-8f2b-ce91e3febf5e",
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "      <th>X - X_mean</th>\n",
       "      <th>y - y_mean</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20</td>\n",
       "      <td>-0.5200</td>\n",
       "      <td>-5.4000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25</td>\n",
       "      <td>0.0800</td>\n",
       "      <td>-0.4000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22</td>\n",
       "      <td>-0.2200</td>\n",
       "      <td>-3.4000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33</td>\n",
       "      <td>0.4800</td>\n",
       "      <td>7.6000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27</td>\n",
       "      <td>0.1800</td>\n",
       "      <td>1.6000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       X   y  X - X_mean  y - y_mean\n",
       "0 8.5000  20     -0.5200     -5.4000\n",
       "1 9.1000  25      0.0800     -0.4000\n",
       "2 8.8000  22     -0.2200     -3.4000\n",
       "3 9.5000  33      0.4800      7.6000\n",
       "4 9.2000  27      0.1800      1.6000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let us take every mouse weight and see how much it is deviated from the average\n",
    "# Calculate y - y_mean for every individual item in y\n",
    "myDataFrame['y - y_mean'] = myDataFrame['y'] - mean_y_value\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "93e49f10-ecdc-4723-822f-a7b665d92eeb",
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "      <th>X - X_mean</th>\n",
       "      <th>y - y_mean</th>\n",
       "      <th>(X - X_mean) * (y - y_mean)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20</td>\n",
       "      <td>-0.5200</td>\n",
       "      <td>-5.4000</td>\n",
       "      <td>2.8080</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25</td>\n",
       "      <td>0.0800</td>\n",
       "      <td>-0.4000</td>\n",
       "      <td>-0.0320</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22</td>\n",
       "      <td>-0.2200</td>\n",
       "      <td>-3.4000</td>\n",
       "      <td>0.7480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33</td>\n",
       "      <td>0.4800</td>\n",
       "      <td>7.6000</td>\n",
       "      <td>3.6480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27</td>\n",
       "      <td>0.1800</td>\n",
       "      <td>1.6000</td>\n",
       "      <td>0.2880</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       X   y  X - X_mean  y - y_mean  (X - X_mean) * (y - y_mean)\n",
       "0 8.5000  20     -0.5200     -5.4000                       2.8080\n",
       "1 9.1000  25      0.0800     -0.4000                      -0.0320\n",
       "2 8.8000  22     -0.2200     -3.4000                       0.7480\n",
       "3 9.5000  33      0.4800      7.6000                       3.6480\n",
       "4 9.2000  27      0.1800      1.6000                       0.2880"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myDataFrame['(X - X_mean) * (y - y_mean)'] = myDataFrame['X - X_mean'] * myDataFrame['y - y_mean']\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d5fe979-6ba5-4765-83c4-30ccf5fe73a2",
   "metadata": {},
   "source": [
    "### Variance\n",
    "\n",
    "Variance is a measure of how much the values in a dataset deviate from the mean. It quantifies the spread or dispersion of the data. The formula for the variance of a set of values X is:\n",
    "\n",
    "<b>Var(X) = Mean((X - X_mean) ** 2)</b><br/>\n",
    "\n",
    "<b>Interpretation of Variance:</b><br/>\n",
    "Variance gives you a numerical value that represents the spread of the data.<br/>\n",
    "<ol>\n",
    "    <li>A higher variance means the data points are more spread out from the mean</li>\n",
    "    <li>A lower variance means the data points are closer to the mean</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "89198449-3d60-439e-aceb-1b12420b3495",
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "      <th>X - X_mean</th>\n",
       "      <th>y - y_mean</th>\n",
       "      <th>(X - X_mean) * (y - y_mean)</th>\n",
       "      <th>(X - X_mean) ^ 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20</td>\n",
       "      <td>-0.5200</td>\n",
       "      <td>-5.4000</td>\n",
       "      <td>2.8080</td>\n",
       "      <td>0.2704</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25</td>\n",
       "      <td>0.0800</td>\n",
       "      <td>-0.4000</td>\n",
       "      <td>-0.0320</td>\n",
       "      <td>0.0064</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22</td>\n",
       "      <td>-0.2200</td>\n",
       "      <td>-3.4000</td>\n",
       "      <td>0.7480</td>\n",
       "      <td>0.0484</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33</td>\n",
       "      <td>0.4800</td>\n",
       "      <td>7.6000</td>\n",
       "      <td>3.6480</td>\n",
       "      <td>0.2304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27</td>\n",
       "      <td>0.1800</td>\n",
       "      <td>1.6000</td>\n",
       "      <td>0.2880</td>\n",
       "      <td>0.0324</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       X   y  X - X_mean  y - y_mean  (X - X_mean) * (y - y_mean)  \\\n",
       "0 8.5000  20     -0.5200     -5.4000                       2.8080   \n",
       "1 9.1000  25      0.0800     -0.4000                      -0.0320   \n",
       "2 8.8000  22     -0.2200     -3.4000                       0.7480   \n",
       "3 9.5000  33      0.4800      7.6000                       3.6480   \n",
       "4 9.2000  27      0.1800      1.6000                       0.2880   \n",
       "\n",
       "   (X - X_mean) ^ 2  \n",
       "0            0.2704  \n",
       "1            0.0064  \n",
       "2            0.0484  \n",
       "3            0.2304  \n",
       "4            0.0324  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "myDataFrame['(X - X_mean) ^ 2'] = myDataFrame['X - X_mean'] * myDataFrame['X - X_mean']\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a492fa62-4326-4555-bc33-65faf518fa74",
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
       "      <th>X</th>\n",
       "      <th>y</th>\n",
       "      <th>X - X_mean</th>\n",
       "      <th>y - y_mean</th>\n",
       "      <th>(X - X_mean) * (y - y_mean)</th>\n",
       "      <th>(X - X_mean) ^ 2</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8.5000</td>\n",
       "      <td>20.0000</td>\n",
       "      <td>-0.5200</td>\n",
       "      <td>-5.4000</td>\n",
       "      <td>2.8080</td>\n",
       "      <td>0.2704</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>9.1000</td>\n",
       "      <td>25.0000</td>\n",
       "      <td>0.0800</td>\n",
       "      <td>-0.4000</td>\n",
       "      <td>-0.0320</td>\n",
       "      <td>0.0064</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.8000</td>\n",
       "      <td>22.0000</td>\n",
       "      <td>-0.2200</td>\n",
       "      <td>-3.4000</td>\n",
       "      <td>0.7480</td>\n",
       "      <td>0.0484</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>9.5000</td>\n",
       "      <td>33.0000</td>\n",
       "      <td>0.4800</td>\n",
       "      <td>7.6000</td>\n",
       "      <td>3.6480</td>\n",
       "      <td>0.2304</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9.2000</td>\n",
       "      <td>27.0000</td>\n",
       "      <td>0.1800</td>\n",
       "      <td>1.6000</td>\n",
       "      <td>0.2880</td>\n",
       "      <td>0.0324</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>MEAN</th>\n",
       "      <td>9.0200</td>\n",
       "      <td>25.4000</td>\n",
       "      <td>-0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>1.4920</td>\n",
       "      <td>0.1176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SUM</th>\n",
       "      <td>45.1000</td>\n",
       "      <td>127.0000</td>\n",
       "      <td>-0.0000</td>\n",
       "      <td>0.0000</td>\n",
       "      <td>7.4600</td>\n",
       "      <td>0.5880</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           X        y  X - X_mean  y - y_mean  (X - X_mean) * (y - y_mean)  \\\n",
       "0     8.5000  20.0000     -0.5200     -5.4000                       2.8080   \n",
       "1     9.1000  25.0000      0.0800     -0.4000                      -0.0320   \n",
       "2     8.8000  22.0000     -0.2200     -3.4000                       0.7480   \n",
       "3     9.5000  33.0000      0.4800      7.6000                       3.6480   \n",
       "4     9.2000  27.0000      0.1800      1.6000                       0.2880   \n",
       "MEAN  9.0200  25.4000     -0.0000      0.0000                       1.4920   \n",
       "SUM  45.1000 127.0000     -0.0000      0.0000                       7.4600   \n",
       "\n",
       "      (X - X_mean) ^ 2  \n",
       "0               0.2704  \n",
       "1               0.0064  \n",
       "2               0.0484  \n",
       "3               0.2304  \n",
       "4               0.0324  \n",
       "MEAN            0.1176  \n",
       "SUM             0.5880  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let us add a new row MEAN and SUM\n",
    "myDataFrame.loc['MEAN'] = myDataFrame.iloc[0:5].mean()\n",
    "myDataFrame.loc['SUM'] = myDataFrame.iloc[0:5].sum()\n",
    "myDataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a3800fa7-1259-4db6-a291-fa0d03ca4235",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Covariance: 1.4919999999999993\n"
     ]
    }
   ],
   "source": [
    "covariance = myDataFrame.loc['MEAN', '(X - X_mean) * (y - y_mean)']\n",
    "print(\"Covariance:\", covariance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "603ce197-c79e-4bfb-84f6-3af199a41600",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Covariance: [[ 0.1176  1.492 ]\n",
      " [ 1.492  20.24  ]]\n"
     ]
    }
   ],
   "source": [
    "covariance_by_np = np.cov(X, y, ddof=0)\n",
    "print(\"Covariance:\", covariance_by_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7fa6674f-2b56-4f8d-869f-0659fd68c8f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Variance: 0.11759999999999986\n"
     ]
    }
   ],
   "source": [
    "variance = myDataFrame.loc['MEAN', '(X - X_mean) ^ 2']\n",
    "print(\"Variance:\", variance)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "16ec4487-c21a-4d82-874d-095d2838eb97",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Covariance: 0.11759999999999986\n"
     ]
    }
   ],
   "source": [
    "variance_by_np = np.var(myDataFrame.iloc[0:5]['X'], ddof=0)\n",
    "print(\"Covariance:\", variance_by_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "01f655a0-f049-42f8-bb39-afd2b3b57e7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Covariance: 20.24\n"
     ]
    }
   ],
   "source": [
    "variance_by_np = np.var(myDataFrame.iloc[0:5]['y'], ddof=0)\n",
    "print(\"Covariance:\", variance_by_np)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3c5765b2-52b7-4aeb-a5df-13d807068132",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(12.687074829931982)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "slope = covariance / variance\n",
    "slope"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c37343c2-dd47-418b-8e0c-1a82d85a0017",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(-89.0374149659865)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# y = mx + c\n",
    "coefficient = (myDataFrame.loc['SUM', 'y'] - (slope * myDataFrame.loc['SUM', 'X'])) / 5\n",
    "coefficient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "83907145-0466-4a30-9de1-1b3cf285a6a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12.68707482993198\n",
      "-89.03741496598641\n"
     ]
    }
   ],
   "source": [
    "polyfit_slope, polyfit_coefficient = np.polyfit(X, y, 1)\n",
    "print(polyfit_slope)\n",
    "print(polyfit_coefficient)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "92b6108c-a73d-48ca-ab16-dcf2faf73f9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_fit = np.linspace(np.min(X), np.max(X), 100)\n",
    "y_fit = (slope * x_fit) + coefficient"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b453209d-9da1-4ebf-bf8f-8e3e2967ec89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x263a4b54500>]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(4,3))\n",
    "plt.scatter(X, y, color='blue')\n",
    "plt.plot(x_fit, y_fit)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2c45a3b-af45-4588-8f61-ad00b706b012",
   "metadata": {},
   "source": [
    "### Calculating and plotting Residuals"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "daf87bc5-a153-407e-9597-521a9bf41725",
   "metadata": {},
   "source": [
    "#### Calculate and the predict y value for each x value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9b151802-8547-4fce-aea7-b197adbff87b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([18.80272109, 26.41496599, 22.60884354, 31.48979592, 27.68367347])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Let us calculate the predicted values\n",
    "y_pred = slope * X + coefficient\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "71af33c0-ac25-4a7b-bd15-0395d4d6de54",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x263a4b8d5b0>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the actual value in green round marker vs predicted value in blue x marker\n",
    "# show the residuals line in red color\n",
    "plt.figure(figsize=(4,3))\n",
    "plt.plot(x_fit, y_fit, color='black')\n",
    "for x1, y1, p1 in zip(X, y, y_pred):\n",
    "    plt.plot([x1, x1], [y1, p1], color='red', marker='x', markersize=5, markevery=[1, 1], markerfacecolor='blue', markeredgecolor='blue')\n",
    "plt.scatter(X, y, color='green', marker='o', s=15)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b83a0801-f85c-4398-bc23-0fa2510fec1f",
   "metadata": {},
   "source": [
    "#### Calculating R squared value from the residual values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "b233b5f5-70d9-4431-97ea-7aa61cd326ea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1.19727891, -1.41496599, -0.60884354,  1.51020408, -0.68367347])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "residuals = y - y_pred\n",
    "residuals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "b30ab443-aee3-43fc-8216-4411e4026bf9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(6.554421768707507)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# sum of square of residuals of best fitted line\n",
    "ss_residuals = np.sum(residuals ** 2)\n",
    "ss_residuals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "2fea44a4-618f-4ffe-8cea-3c0cc2454254",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(101.19999999999999)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_y = np.mean(y)\n",
    "average_residuals = y - mean_y\n",
    "ss_tot = np.sum(average_residuals ** 2)\n",
    "ss_tot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "bff9077d-bb31-4150-989a-c44b8db404d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.float64(0.9352329864752222)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r_squared_value = 1 - (ss_residuals/ss_tot)\n",
    "r_squared_value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "48e0cb1b-f83e-46bc-896d-1de76ef7c6b0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9352329864752222"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r_squared = r2_score(y, y_pred)\n",
    "r_squared"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac72c56f-78db-423d-ad53-b1f2b710c9c5",
   "metadata": {},
   "source": [
    "#### Homoscedasticity of Errors:\n",
    "<ol>\n",
    "    <li>In a regression model, we try to predict something (like mouse weight) based on some input (like mouse height).</li>\n",
    "    <li>The difference between the actual values and the predicted values is called errors or residuals.</li>\n",
    "    <li>Homoscedasticity means that no matter what the predicted value is, the size of the errors stays roughly the same.</li>\n",
    "    <li>If the errors are not spread out evenly (i.e., heteroscedasticity exists), your model might be making better predictions for some values and worse predictions for others. This can make your model unreliable, especially when you try to make predictions for new data.</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d5380260-7d05-4c94-8d66-d027a250c666",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1.19727891, -1.41496599, -0.60884354,  1.51020408, -0.68367347])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "residuals = y - y_pred\n",
    "residuals"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "a87d28fa-da06-4a0b-9cc6-ed6c9e711cfe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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jPT0dAQEBJT779yz+Wpj8HfOGBWNSldOTo507txZWrRooW5bvXq1WFlZFYh97bXXZOLEiaWZXgHa8n3W7du3pVq1ajJlypRSzEw3XTkPHz5chgwZon6McrLHoS3fqKgoASAuLi6ybNkyOX78uIwdO1asrKzkwoULZZit7tf33r170r59ewEgFhYW4ujoKH/++WcZZZnn1KlTYmdnJ+bm5uLk5CS//fabiJTf75yufJ/1PN+58vGvHell1KhRiImJKTf/ORalqHxTU1PRuXNn+Pv7Y/r06aWbnA7act66dSt27dpVZv3thdGWb25uLgDgvffew+DBgwEAjRo1ws6dO7Fs2TKEh4eXSa6A7s/Ep59+ivv37+Ovv/6Cq6srNm/ejN69e2P//v2oV69emeRau3ZtREdHQ6VSYcOGDRg4cCD27t1bJrkUh658/f391THP/Z173upGpWvUqFFStWpVuXTpkkb7zp07BYDcu3dPo71atWoye/bsUsxQk65886WmpkpAQIAEBwfLw4cPSzk77XTlPGbMGFEoFGJubq6eAIiZmZkEBQWVTbKiO99Lly4JAFm5cqVGe+/evaV///6lmaIGXfnGxcUJAImJidFoDw4Olvfee680UyxUcHCwDB8+vNx+556Vn28+Y3znOMZhIkQEH3zwATZt2oRdu3bBz89PY36TJk1gaWmJnTt3qttiY2Nx9epVnf2bJamofIG8/3rat28PKysrbN26FdbW1qWe59OKynnSpEk4deoUoqOj1RMAzJkzB8uXLy93+fr6+sLb27vAIa8XLlxA9erVSzNVAEXnm5GRAQAFjvgyNzdX7z2VB7m5ucjKyip33zld8vMFjPidM0JBo1IwYsQIcXJykj179khSUpJ6ysjIUMe8//77Uq1aNdm1a5ccPXpUAgICJCAgoFzmq1KppHnz5lKvXj2Ji4vTiHn8+HG5zFkblOEYR3HynTNnjjg6Osr69evl4sWL8sknn4i1tbXExcWVu3yzs7OlVq1a0rJlSzl06JDExcXJ119/LQqFQmc/fUmbNGmS7N27VxISEuTUqVMyadIkUSgUsn37dhEpX9+5ovI15neOhcNEQMc9g/PvsS4i8vDhQxk5cqRUqlRJbG1tpUePHpKUlFQu8929e7fOmISEhHKZs65lyqpwFDff8PBwqVq1qtja2kpAQIDs37+/3OZ7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",
      "text/plain": [
       "<Figure size 400x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(4,3))\n",
    "plt.scatter(y_pred, residuals)\n",
    "plt.axhline(y=0, color='r', linestyle='--')\n",
    "plt.xlabel('Fitted values (Predicted Y)')\n",
    "plt.ylabel('Residuals')\n",
    "plt.title('Residuals vs Fitted Values')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8347365-c12b-4d20-8596-8eca7b4552a5",
   "metadata": {},
   "source": [
    "<b>Homoscedasticity Interpretation:</b><br/>\n",
    "<ol>\n",
    "    <li>If the residuals are randomly scattered around 0 with no clear pattern (such as a cone or funnel shape), the assumption of homoscedasticity holds.</li>\n",
    "    <li>Heteroscedasticity: If the residuals form a funnel shape (either narrowing or widening as fitted values increase), this indicates heteroscedasticity (i.e., the variance of the residuals changes with the predicted values).</li>\n",
    "</ol>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1fab641-b910-4711-8685-13aa5b040603",
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
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
