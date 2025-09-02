import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create reports directory if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

# Set style for visualizations
plt.style.use('ggplot')
sns.set_palette("husl")

class OutputLogger:
    """Class to capture all console output and save to file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

class CustomerDataAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the analyzer with the data file path
        """
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.report_content = []
        
    def _add_to_report(self, content):
        """Add content to the report"""
        self.report_content.append(content)
        
    def load_and_clean_data(self):
        """
        Load and clean the customer data
        """
        # Load the data
        self.df = pd.read_csv(self.file_path)
        
        # Display basic info
        print("Data Overview:")
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Handle missing values if any
        if self.df.isnull().sum().sum() > 0:
            # Fill missing Age Group with mode
            if 'Age Group' in self.df.columns:
                self.df['Age Group'].fillna(self.df['Age Group'].mode()[0], inplace=True)
            
            # Fill numerical columns with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Check data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Add to report
        self._add_to_report("Data Overview:")
        self._add_to_report(f"Dataset shape: {self.df.shape}")
        self._add_to_report("\nFirst 5 rows:")
        self._add_to_report(str(self.df.head()))
        self._add_to_report("\nMissing values:")
        self._add_to_report(str(self.df.isnull().sum()))
        self._add_to_report("\nData types:")
        self._add_to_report(str(self.df.dtypes))
        
        return self.df
    
    def perform_demographic_analysis(self):
        """
        Analyze demographic patterns in the data
        """
        print("\n" + "="*50)
        print("DEMOGRAPHIC ANALYSIS")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("DEMOGRAPHIC ANALYSIS")
        self._add_to_report("="*50)
        
        # Gender distribution
        gender_counts = self.df['Gender'].value_counts()
        print("\nGender Distribution:")
        print(gender_counts)
        
        # Age group distribution
        if 'Age Group' in self.df.columns:
            age_group_counts = self.df['Age Group'].value_counts()
            print("\nAge Group Distribution:")
            print(age_group_counts)
        
        # Income distribution
        print("\nAnnual Income Statistics (k$):")
        print(self.df['Annual Income (k$)'].describe())
        
        # Spending score distribution
        print("\nSpending Score Statistics:")
        print(self.df['Spending Score (1-100)'].describe())
        
        # Add to report
        self._add_to_report("\nGender Distribution:")
        self._add_to_report(str(gender_counts))
        
        if 'Age Group' in self.df.columns:
            self._add_to_report("\nAge Group Distribution:")
            self._add_to_report(str(age_group_counts))
        
        self._add_to_report("\nAnnual Income Statistics (k$):")
        self._add_to_report(str(self.df['Annual Income (k$)'].describe()))
        
        self._add_to_report("\nSpending Score Statistics:")
        self._add_to_report(str(self.df['Spending Score (1-100)'].describe()))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Demographic Analysis', fontsize=16)
        
        # Gender distribution pie chart
        gender_counts.plot.pie(autopct='%1.1f%%', ax=axes[0, 0])
        axes[0, 0].set_title('Gender Distribution')
        axes[0, 0].set_ylabel('')
        
        # Age group distribution
        if 'Age Group' in self.df.columns:
            age_order = ['18-25', '26-35', '36-50', '51-65', '65+']
            age_group_counts = self.df['Age Group'].value_counts()
            age_group_counts = age_group_counts.reindex(age_order, fill_value=0)
            age_group_counts.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Age Group Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Income distribution histogram
        self.df['Annual Income (k$)'].plot(kind='hist', bins=20, ax=axes[1, 0])
        axes[1, 0].set_title('Annual Income Distribution')
        axes[1, 0].set_xlabel('Annual Income (k$)')
        
        # Spending score distribution histogram
        self.df['Spending Score (1-100)'].plot(kind='hist', bins=20, ax=axes[1, 1])
        axes[1, 1].set_title('Spending Score Distribution')
        axes[1, 1].set_xlabel('Spending Score')
        
        plt.tight_layout()
        plt.savefig('reports/demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'gender_distribution': gender_counts.to_dict(),
            'age_group_distribution': age_group_counts.to_dict() if 'Age Group' in self.df.columns else None,
            'income_stats': self.df['Annual Income (k$)'].describe().to_dict(),
            'spending_stats': self.df['Spending Score (1-100)'].describe().to_dict()
        }
    
    def perform_segmentation_analysis(self):
        """
        Perform customer segmentation using clustering
        """
        print("\n" + "="*50)
        print("CUSTOMER SEGMENTATION ANALYSIS")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("CUSTOMER SEGMENTATION ANALYSIS")
        self._add_to_report("="*50)
        
        # Prepare data for clustering
        cluster_data = self.df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].copy()
        
        # Standardize the data
        cluster_data_scaled = self.scaler.fit_transform(cluster_data)
        
        # Find optimal k using elbow method
        wcss = []
        k_range = range(1, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_data_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, wcss, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('reports/elbow_method.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Based on elbow method, choose k=5 (you can adjust this)
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.df['Segment'] = kmeans.fit_predict(cluster_data_scaled)
        
        # Analyze segments
        segment_analysis = self.df.groupby('Segment').agg({
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std'],
            'Age': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        print("\nSegment Analysis:")
        print(segment_analysis)
        
        # Add to report
        self._add_to_report("\nSegment Analysis:")
        self._add_to_report(str(segment_analysis))
        
        # Visualize segments
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.df['Annual Income (k$)'], 
            self.df['Spending Score (1-100)'], 
            c=self.df['Segment'], 
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Segment')
        plt.title('Customer Segmentation by Income and Spending')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.savefig('reports/customer_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3D visualization of segments
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.df['Annual Income (k$)'],
            self.df['Spending Score (1-100)'],
            self.df['Age'],
            c=self.df['Segment'],
            cmap='viridis',
            alpha=0.7
        )
        
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score')
        ax.set_zlabel('Age')
        ax.set_title('3D Customer Segmentation')
        plt.colorbar(scatter, label='Segment')
        plt.savefig('reports/3d_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'segments': segment_analysis.to_dict(),
            'segment_labels': self.df['Segment'].value_counts().to_dict()
        }
    
    def perform_correlation_analysis(self):
        """
        Analyze correlations between different variables
        """
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("CORRELATION ANALYSIS")
        self._add_to_report("="*50)
        
        # Select numerical columns for correlation
        numerical_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 
                          'Estimated Savings (k$)', 'Credit Score', 'Loyalty Years']
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        # Add to report
        self._add_to_report("\nCorrelation Matrix:")
        self._add_to_report(str(corr_matrix))
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Pairplot to visualize relationships
        sns.pairplot(self.df[numerical_cols])
        plt.suptitle('Pairplot of Numerical Variables', y=1.02)
        plt.savefig('reports/pairplot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix.to_dict()
    
    def perform_rfm_analysis(self):
        """
        Perform RFM analysis (adapted for available data)
        """
        print("\n" + "="*50)
        print("RFM ANALYSIS (ADAPTED)")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("RFM ANALYSIS (ADAPTED)")
        self._add_to_report("="*50)
        
        # Create RFM segments
        # Since we don't have recency data, we'll adapt:
        # Monetary: Annual Income + Spending Score
        # Frequency: Loyalty Years
        # We'll use Credit Score as a proxy for recency risk
        
        # Calculate RFM values
        self.df['RFM_Monetary'] = (self.df['Annual Income (k$)'] + self.df['Spending Score (1-100)']) / 2
        self.df['RFM_Frequency'] = self.df['Loyalty Years']
        self.df['RFM_Recency'] = self.df['Credit Score']  # Higher credit score = lower recency risk
        
        # Create RFM scores (1-5) with error handling for duplicate bins
        try:
            self.df['M_Score'] = pd.qcut(self.df['RFM_Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
        except:
            # If qcut fails, use cut with custom bins
            monetary_bins = pd.cut(self.df['RFM_Monetary'], bins=5, labels=[1, 2, 3, 4, 5])
            self.df['M_Score'] = monetary_bins.cat.codes + 1
        
        try:
            self.df['F_Score'] = pd.qcut(self.df['RFM_Frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
        except:
            # If qcut fails, use cut with custom bins
            frequency_bins = pd.cut(self.df['RFM_Frequency'], bins=5, labels=[1, 2, 3, 4, 5])
            self.df['F_Score'] = frequency_bins.cat.codes + 1
        
        try:
            self.df['R_Score'] = pd.qcut(self.df['RFM_Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
        except:
            # If qcut fails, use cut with custom bins (reverse order for recency)
            recency_bins = pd.cut(self.df['RFM_Recency'], bins=5, labels=[5, 4, 3, 2, 1])
            self.df['R_Score'] = recency_bins.cat.codes + 1
            # Reverse the scores since higher credit score should give higher R_Score
            self.df['R_Score'] = 6 - self.df['R_Score']
        
        # Combine scores
        self.df['RFM_Score'] = self.df['R_Score'] + self.df['F_Score'] + self.df['M_Score']
        
        # Create segments based on RFM score
        def get_rfm_segment(score):
            if score >= 12:
                return 'Champions'
            elif score >= 9:
                return 'Loyal Customers'
            elif score >= 7:
                return 'Potential Loyalists'
            elif score >= 5:
                return 'At Risk'
            else:
                return 'Lost Customers'
        
        self.df['RFM_Segment'] = self.df['RFM_Score'].apply(get_rfm_segment)
        
        # Analyze RFM segments
        rfm_analysis = self.df.groupby('RFM_Segment').agg({
            'RFM_Score': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'Loyalty Years': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        print("\nRFM Segment Analysis:")
        print(rfm_analysis)
        
        # Add to report
        self._add_to_report("\nRFM Segment Analysis:")
        self._add_to_report(str(rfm_analysis))
        
        # Visualize RFM segments
        plt.figure(figsize=(10, 6))
        rfm_analysis['CustomerID'].plot(kind='bar')
        plt.title('Customer Distribution by RFM Segment')
        plt.xlabel('RFM Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        plt.savefig('reports/rfm_segments.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return rfm_analysis.to_dict()
    
    def perform_preference_analysis(self):
        """
        Analyze customer preferences by category
        """
        print("\n" + "="*50)
        print("PREFERENCE ANALYSIS")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("PREFERENCE ANALYSIS")
        self._add_to_report("="*50)
        
        if 'Preferred Category' not in self.df.columns:
            print("No Preferred Category data available")
            self._add_to_report("No Preferred Category data available")
            return None
        
        # Analyze preferences by demographic factors
        category_counts = self.df['Preferred Category'].value_counts()
        print("\nPreferred Category Distribution:")
        print(category_counts)
        
        # Preferences by gender
        gender_pref = pd.crosstab(self.df['Gender'], self.df['Preferred Category'], normalize='index') * 100
        print("\nPreferences by Gender (%):")
        print(gender_pref.round(2))
        
        # Preferences by age group
        if 'Age Group' in self.df.columns:
            age_pref = pd.crosstab(self.df['Age Group'], self.df['Preferred Category'], normalize='index') * 100
            print("\nPreferences by Age Group (%):")
            print(age_pref.round(2))
        
        # Preferences by income quartile
        self.df['Income_Quartile'] = pd.qcut(self.df['Annual Income (k$)'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        income_pref = pd.crosstab(self.df['Income_Quartile'], self.df['Preferred Category'], normalize='index') * 100
        print("\nPreferences by Income Quartile (%):")
        print(income_pref.round(2))
        
        # Add to report
        self._add_to_report("\nPreferred Category Distribution:")
        self._add_to_report(str(category_counts))
        self._add_to_report("\nPreferences by Gender (%):")
        self._add_to_report(str(gender_pref.round(2)))
        
        if 'Age Group' in self.df.columns:
            self._add_to_report("\nPreferences by Age Group (%):")
            self._add_to_report(str(age_pref.round(2)))
        
        self._add_to_report("\nPreferences by Income Quartile (%):")
        self._add_to_report(str(income_pref.round(2)))
        
        # Visualize preferences
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Customer Preference Analysis', fontsize=16)
        
        # Category distribution
        category_counts.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Preferred Category Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Preferences by gender
        gender_pref.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Preferences by Gender')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Preferences by age group
        if 'Age Group' in self.df.columns:
            age_pref.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Preferences by Age Group')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Preferences by income quartile
        income_pref.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Preferences by Income Quartile')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('reports/preference_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'category_distribution': category_counts.to_dict(),
            'gender_preferences': gender_pref.to_dict(),
            'income_preferences': income_pref.to_dict()
        }
    
    def predict_spending_score(self):
        """
        Build a model to predict spending score
        """
        print("\n" + "="*50)
        print("SPENDING SCORE PREDICTION")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("SPENDING SCORE PREDICTION")
        self._add_to_report("="*50)
        
        # Prepare features and target
        features = ['Age', 'Annual Income (k$)', 'Estimated Savings (k$)', 'Credit Score', 'Loyalty Years']
        features = [f for f in features if f in self.df.columns]
        
        X = self.df[features]
        y = self.df['Spending Score (1-100)']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("\nFeature Importance:")
        print(importance)
        
        # Add to report
        self._add_to_report(f"R² Score: {r2:.4f}")
        self._add_to_report(f"RMSE: {rmse:.4f}")
        self._add_to_report("\nFeature Importance:")
        self._add_to_report(str(importance))
        
        # Visualize predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Spending Score')
        plt.ylabel('Predicted Spending Score')
        plt.title('Actual vs Predicted Spending Score')
        plt.savefig('reports/spending_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': importance.to_dict('records')
        }
    
    def generate_summary_report(self, analyses):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("SUMMARY REPORT")
        self._add_to_report("="*50)
        
        # Key insights
        total_customers = len(self.df)
        avg_income = self.df['Annual Income (k$)'].mean()
        avg_spending = self.df['Spending Score (1-100)'].mean()
        
        print(f"Total Customers: {total_customers}")
        print(f"Average Annual Income: ${avg_income:.2f}k")
        print(f"Average Spending Score: {avg_spending:.2f}/100")
        
        # Top segments
        if 'segment_labels' in analyses['segmentation']:
            largest_segment = max(analyses['segmentation']['segment_labels'], 
                                 key=analyses['segmentation']['segment_labels'].get)
            print(f"Largest Customer Segment: {largest_segment} "
                  f"({analyses['segmentation']['segment_labels'][largest_segment]} customers)")
        
        # Top RFM segment
        if 'rfm' in analyses and analyses['rfm']:
            rfm_segment_counts = self.df['RFM_Segment'].value_counts()
            top_rfm_segment = rfm_segment_counts.idxmax()
            print(f"Top RFM Segment: {top_rfm_segment} ({rfm_segment_counts[top_rfm_segment]} customers)")
        
        # Key correlations
        if 'correlation' in analyses:
            corr_matrix = analyses['correlation']
            high_corr = []
            for feature, correlations in corr_matrix.items():
                for other_feature, value in correlations.items():
                    if feature != other_feature and abs(value) > 0.5:
                        high_corr.append((feature, other_feature, value))
            
            if high_corr:
                print("\nStrong Correlations (> |0.5|):")
                for corr in high_corr[:5]:  # Show top 5
                    print(f"{corr[0]} - {corr[1]}: {corr[2]:.3f}")
        
        # Add to report
        self._add_to_report(f"Total Customers: {total_customers}")
        self._add_to_report(f"Average Annual Income: ${avg_income:.2f}k")
        self._add_to_report(f"Average Spending Score: {avg_spending:.2f}/100")
        
        if 'segment_labels' in analyses['segmentation']:
            self._add_to_report(f"Largest Customer Segment: {largest_segment} "
                              f"({analyses['segmentation']['segment_labels'][largest_segment]} customers)")
        
        if 'rfm' in analyses and analyses['rfm']:
            self._add_to_report(f"Top RFM Segment: {top_rfm_segment} ({rfm_segment_counts[top_rfm_segment]} customers)")
        
        if high_corr:
            self._add_to_report("\nStrong Correlations (> |0.5|):")
            for corr in high_corr[:5]:
                self._add_to_report(f"{corr[0]} - {corr[1]}: {corr[2]:.3f}")
        
        # Business recommendations
        print("\n" + "="*50)
        print("BUSINESS RECOMMENDATIONS")
        print("="*50)
        
        # Add to report
        self._add_to_report("\n" + "="*50)
        self._add_to_report("BUSINESS RECOMMENDATIONS")
        self._add_to_report("="*50)
        
        # Recommendation based on segments
        if 'segmentation' in analyses:
            print("\n1. Segment-Based Recommendations:")
            self._add_to_report("\n1. Segment-Based Recommendations:")
            
            segment_means = self.df.groupby('Segment').agg({
                'Annual Income (k$)': 'mean',
                'Spending Score (1-100)': 'mean'
            })
            
            for segment in segment_means.index:
                income = segment_means.loc[segment, 'Annual Income (k$)']
                spending = segment_means.loc[segment, 'Spending Score (1-100)']
                size = analyses['segmentation']['segment_labels'][segment]
                
                if spending > 70 and income > 70:
                    rec = f"   - Segment {segment}: High-value customers ({size} people). " \
                          "Offer premium products and loyalty rewards."
                elif spending > 70 and income < 50:
                    rec = f"   - Segment {segment}: Big spenders with moderate income ({size} people). " \
                          "Offer value-based products and payment plans."
                elif spending < 30 and income > 70:
                    rec = f"   - Segment {segment}: High income but low spending ({size} people). " \
                          "Investigate reasons for low engagement and create targeted campaigns."
                else:
                    rec = f"   - Segment {segment}: Typical customers ({size} people). " \
                          "Focus on retention and cross-selling."
                
                print(rec)
                self._add_to_report(rec)
        
        # Recommendation based on RFM
        if 'rfm' in analyses and analyses['rfm']:
            print("\n2. RFM-Based Recommendations:")
            self._add_to_report("\n2. RFM-Based Recommendations:")
            
            for segment in self.df['RFM_Segment'].unique():
                count = (self.df['RFM_Segment'] == segment).sum()
                if segment == 'Champions':
                    rec = f"   - {segment} ({count} people): Reward these customers. " \
                          "They are your most valuable customers."
                elif segment == 'Loyal Customers':
                    rec = f"   - {segment} ({count} people): Upsell higher value products. " \
                          "Ask for reviews and engage them."
                elif segment == 'Potential Loyalists':
                    rec = f"   - {segment} ({count} people): Offer membership or loyalty programs. " \
                          "They have potential to become loyal customers."
                elif segment == 'At Risk':
                    rec = f"   - {segment} ({count} people): Send personalized offers. " \
                          "Reconnect with them and win back."
                elif segment == 'Lost Customers':
                    rec = f"   - {segment} ({count} people): Revive interest with reachout campaigns. " \
                          "Offer discounts and ask for feedback."
                
                print(rec)
                self._add_to_report(rec)
        
        # Recommendation based on preferences
        if 'preferences' in analyses and analyses['preferences']:
            print("\n3. Preference-Based Recommendations:")
            self._add_to_report("\n3. Preference-Based Recommendations:")
            
            prefs = analyses['preferences']['category_distribution']
            top_category = max(prefs, key=prefs.get)
            rec1 = f"   - Most popular category: {top_category} ({prefs[top_category]} customers). " \
                  "Leverage this category for cross-selling and promotions."
            
            print(rec1)
            self._add_to_report(rec1)
            
            # Find underutilized categories
            if len(prefs) > 1:
                bottom_category = min(prefs, key=prefs.get)
                rec2 = f"   - Least popular category: {bottom_category} ({prefs[bottom_category]} customers). " \
                      "Investigate why this category is underperforming and consider promotions."
                
                print(rec2)
                self._add_to_report(rec2)
    
    def export_report_to_file(self):
        """Export the report content to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"reports/customer_analysis_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("CUSTOMER DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for line in self.report_content:
                f.write(line + "\n")
        
        print(f"\nReport exported to: {report_filename}")
        return report_filename
    
    def run_complete_analysis(self):
        """
        Run all analyses and generate comprehensive report
        """
        print("CUSTOMER DATA ANALYSIS SYSTEM")
        print("="*50)
        
        # Add to report
        self._add_to_report("CUSTOMER DATA ANALYSIS SYSTEM")
        self._add_to_report("="*50)
        
        # Load and clean data
        self.load_and_clean_data()
        
        # Perform all analyses
        analyses = {}
        
        analyses['demographic'] = self.perform_demographic_analysis()
        analyses['segmentation'] = self.perform_segmentation_analysis()
        analyses['correlation'] = self.perform_correlation_analysis()
        analyses['rfm'] = self.perform_rfm_analysis()
        analyses['preferences'] = self.perform_preference_analysis()
        analyses['prediction'] = self.predict_spending_score()
        
        # Generate summary report
        self.generate_summary_report(analyses)
        
        # Save analyzed data to Excel in reports folder
        self.df.to_excel('reports/analyzed_customer_data.xlsx', index=False)
        print("\nAnalyzed data saved to 'reports/analyzed_customer_data.xlsx'")
        self._add_to_report("\nAnalyzed data saved to 'reports/analyzed_customer_data.xlsx'")
        
        # Export report to file
        report_file = self.export_report_to_file()
        
        return analyses, report_file

# Main execution
if __name__ == "__main__":
    # Initialize analyzer with your CSV file path
    analyzer = CustomerDataAnalyzer('customer_data.csv')  # Replace with your file path
    
    # Create output logger to capture all console output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"reports/console_output_{timestamp}.txt"
    logger = OutputLogger(log_filename)
    sys.stdout = logger
    
    try:
        # Run complete analysis
        results, report_file = analyzer.run_complete_analysis()
        print(f"\nConsole output saved to: {log_filename}")
        print(f"Analysis report saved to: {report_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Restore stdout
        sys.stdout = logger.terminal
        logger.close()