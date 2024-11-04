import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Note: This is NOT final results, I need better commenting as well as an expanded dataset


# Data Loading and Initial Exploration
def load_and_explore_data(file_path):
    """
    Load the fitness tracker dataset and perform initial exploration
    """
    print("Loading and exploring data...")
    df = pd.read_csv(file_path)
    
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

# Data Preprocessing
def preprocess_data(df):
    """
    Clean and prepare data for analysis
    """
    print("\nPreprocessing data...")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Remove any rows with missing values in our main variables of interest
    df_clean = df.dropna(subset=['calories_burned', 'sleep_hours', 'steps'])
    
    return df_clean

# Data Analysis and Visualization
def analyze_data(df):
    """
    Perform exploratory data analysis and create visualizations
    """
    print("\nPerforming data analysis...")
    
    # Create correlation matrix
    correlation_matrix = df[['calories_burned', 'sleep_hours', 'steps', 'active_minutes', 'heart_rate_avg']].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Fitness Variables')
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Calories Burned vs Sleep Hours
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='calories_burned', y='sleep_hours', alpha=0.5)
    plt.title('Relationship between Calories Burned and Sleep Hours')
    plt.xlabel('Calories Burned')
    plt.ylabel('Sleep Hours')
    plt.show()
    
    # Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.histplot(data=df, x='calories_burned', bins=30, ax=ax1)
    ax1.set_title('Distribution of Calories Burned')
    
    sns.histplot(data=df, x='sleep_hours', bins=30, ax=ax2)
    ax2.set_title('Distribution of Sleep Hours')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Sleep patterns by mood
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='mood', y='sleep_hours')
    plt.title('Sleep Hours Distribution by Mood')
    plt.xticks(rotation=45)
    plt.show()
    
    return correlation_matrix

# Model Training and Evaluation
def train_model(df):
    """
    Train and evaluate linear regression model
    """
    print("\nTraining linear regression model...")
    
    # Prepare features and target
    X = df[['calories_burned']]
    y = df['sleep_hours']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Coefficient (slope): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Visualization of the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel('Calories Burned')
    plt.ylabel('Sleep Hours')
    plt.title('Linear Regression: Calories Burned vs Sleep Hours')
    plt.legend()
    plt.show()
    
    return model, mse, r2

def main():
    """
    Main function to run the analysis
    """
    # Replace with your actual file path
    file_path = "fitness_tracker_dataset.csv"
    
    # Load and explore data
    df = load_and_explore_data(file_path)
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Analyze data
    correlation_matrix = analyze_data(df_clean)
    
    # Train and evaluate model
    model, mse, r2 = train_model(df_clean)
    
    # Additional insights
    print("\nKey Findings:")
    print("1. Correlation between calories burned and sleep hours:", 
          correlation_matrix.loc['calories_burned', 'sleep_hours'].round(4))
    print("2. Average sleep hours:", df_clean['sleep_hours'].mean().round(2))
    print("3. Average calories burned:", df_clean['calories_burned'].mean().round(2))

if __name__ == "__main__":
    main()
