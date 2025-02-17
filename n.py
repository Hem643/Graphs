# # # # # import os
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # # # from sklearn.impute import SimpleImputer
# # # # # from sklearn.ensemble import GradientBoostingRegressor
# # # # # from sklearn.metrics import mean_squared_error, r2_score
# # # # # import matplotlib.pyplot as plt
# # # # # import seaborn as sns
# # # # # import plotly.express as px

# # # # # # Create a directory to save images
# # # # # image_dir = 'images'
# # # # # if not os.path.exists(image_dir):
# # # # #     os.makedirs(image_dir)

# # # # # # Load the CSV file
# # # # # file_path = 'D:/skyward/Opportunity Details By Owner (27).csv'
# # # # # df = pd.read_csv(file_path, skiprows=1)  # Skip the first row which seems to be empty

# # # # # # Display the first few rows
# # # # # print(df.head())

# # # # # # Strip leading and trailing spaces from column names
# # # # # df.columns = df.columns.str.strip()

# # # # # # Check the actual column names
# # # # # print("Column names:", df.columns.tolist())

# # # # # # Data Preprocessing
# # # # # # Drop columns with all missing values
# # # # # df.dropna(axis=1, how='all', inplace=True)

# # # # # # Fill missing values for categorical columns with 'Unknown'
# # # # # categorical_columns = df.select_dtypes(include=['object']).columns
# # # # # df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # # # # Convert date columns to datetime
# # # # # date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # # # # for col in date_columns:
# # # # #     if col in df.columns:
# # # # #         df[col] = pd.to_datetime(df[col], errors='coerce')

# # # # # # Fill missing values for numerical columns with the mean
# # # # # numeric_columns = df.select_dtypes(include=[np.number]).columns
# # # # # imputer = SimpleImputer(strategy='mean')
# # # # # df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # # # # Encode categorical columns
# # # # # label_encoders = {}
# # # # # for col in categorical_columns:
# # # # #     le = LabelEncoder()
# # # # #     df[col] = le.fit_transform(df[col])
# # # # #     label_encoders[col] = le

# # # # # # Scale numerical data
# # # # # scaler = StandardScaler()
# # # # # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # # # # Feature Selection and Target Variable
# # # # # # For this example, let's predict 'Amount'
# # # # # target_column = 'Amount'
# # # # # features = df.drop(columns=[target_column])
# # # # # target = df[target_column]

# # # # # # Train-test split
# # # # # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # # # # Hyperparameter tuning using GridSearchCV
# # # # # param_grid = {
# # # # #     'n_estimators': [50, 100, 200],
# # # # #     'learning_rate': [0.01, 0.1, 0.2],
# # # # #     'max_depth': [3, 4, 5]
# # # # # }

# # # # # grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # # # #                            param_grid=param_grid,
# # # # #                            cv=5,
# # # # #                            scoring='r2',
# # # # #                            n_jobs=-1)

# # # # # grid_search.fit(X_train, y_train)

# # # # # # Best parameters and best score
# # # # # best_params = grid_search.best_params_
# # # # # best_score = grid_search.best_score_
# # # # # print(f"Best Parameters: {best_params}")
# # # # # print(f"Best Cross-Validation R^2 Score: {best_score}")

# # # # # # Train model with best parameters
# # # # # best_model = grid_search.best_estimator_
# # # # # best_model.fit(X_train, y_train)

# # # # # # Make predictions with the best model
# # # # # y_pred_best = best_model.predict(X_test)

# # # # # # Evaluate the best model
# # # # # mse_best = mean_squared_error(y_test, y_pred_best)
# # # # # r2_best = r2_score(y_test, y_pred_best)
# # # # # print(f"Mean Squared Error (Best Model): {mse_best}")
# # # # # print(f"R^2 Score (Best Model): {r2_best}")

# # # # # # Provide insights for new customers
# # # # # feature_importances = best_model.feature_importances_
# # # # # features = X_train.columns
# # # # # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # # # # importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # # # # print("\nTop 5 Features that influence the Amount:")
# # # # # top_5_features = importance_df.head(5)['Feature'].tolist()
# # # # # print(importance_df.head(5))

# # # # # # Decode categorical features back to their original labels
# # # # # for col in categorical_columns:
# # # # #     df[col] = label_encoders[col].inverse_transform(df[col])

# # # # # # Extract only the name from the contact person field
# # # # # df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0])

# # # # # # Create short forms for long names
# # # # # def create_short_form(name, max_length=10):
# # # # #     if len(name) > max_length:
# # # # #         return name[:max_length] + '...'
# # # # #     return name

# # # # # # Apply short forms to specific columns
# # # # # df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # # # # df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # # # # df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # # # # df['Product Short'] = df['Product'].apply(create_short_form)

# # # # # # Plot histograms for Opportunity Name, Territory Name, Contact Person, and Product using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Opportunity Name Short'], kde=True)
# # # # # plt.title('Opportunity Name Distribution')
# # # # # plt.xlabel('Opportunity Name')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_distribution_histogram.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Territory Name Short'], kde=True)
# # # # # plt.title('Territory Name Distribution')
# # # # # plt.xlabel('Territory Name')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_distribution_histogram.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Contact Person Short'], kde=True)
# # # # # plt.title('Contact Person Distribution')
# # # # # plt.xlabel('Contact Person')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'contact_person_distribution_histogram.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Product Short'], kde=True)
# # # # # plt.title('Product Distribution')
# # # # # plt.xlabel('Product')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_distribution_histogram.png'))
# # # # # plt.close()

# # # # # # Plot line graph for Contact Person and Product using short forms
# # # # # contact_person_counts = df['Contact Person Short'].value_counts().sort_index()
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.lineplot(x=contact_person_counts.index, y=contact_person_counts.values)
# # # # # plt.title('Contact Person Line Graph')
# # # # # plt.xlabel('Contact Person')
# # # # # plt.ylabel('Frequency')
# # # # # plt.xticks(rotation=45, ha='right')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'contact_person_line_graph.png'))
# # # # # plt.close()

# # # # # product_counts = df['Product Short'].value_counts().sort_index()
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.lineplot(x=product_counts.index, y=product_counts.values)
# # # # # plt.title('Product Line Graph')
# # # # # plt.xlabel('Product')
# # # # # plt.ylabel('Frequency')
# # # # # plt.xticks(rotation=45, ha='right')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_line_graph.png'))
# # # # # plt.close()

# # # # # # Plot density plots for Opportunity Name, Territory Name, Contact Person, and Product using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Opportunity Name Short'], kde=True)
# # # # # plt.title('Opportunity Name Density')
# # # # # plt.xlabel('Opportunity Name')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_density.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Territory Name Short'], kde=True)
# # # # # plt.title('Territory Name Density')
# # # # # plt.xlabel('Territory Name')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_density.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Contact Person Short'], kde=True)
# # # # # plt.title('Contact Person Density')
# # # # # plt.xlabel('Contact Person')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'contact_person_density.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Product Short'], kde=True)
# # # # # plt.title('Product Density')
# # # # # plt.xlabel('Product')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_density.png'))
# # # # # plt.close()

# # # # # # Plot pie charts for Opportunity Name, Territory Name, Contact Person, and Product using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Opportunity Name Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_distribution_pie.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Territory Name Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_distribution_pie.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Contact Person Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'contact_person_distribution_pie.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Product Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_distribution_pie.png'))
# # # # # plt.close()

# # # # # # Plot pair plots for the top 5 features
# # # # # sns.pairplot(df[top_5_features + [target_column]])
# # # # # plt.suptitle('Pair Plot of Top 5 Features and Amount', y=1.02)
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'pair_plot_top_5_features.png'))
# # # # # plt.close()














# # # # # # import os
# # # # # # import pandas as pd
# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns
# # # # # # import streamlit as st

# # # # # # # --- Streamlit Interface ---
# # # # # # st.title("Opportunity Analysis Dashboard")

# # # # # # # Sidebar for file upload
# # # # # # st.sidebar.header("Data Upload")
# # # # # # uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# # # # # # if uploaded_file is not None:
# # # # # #     # Read the CSV file
# # # # # #     try:
# # # # # #         df = pd.read_csv(uploaded_file, skiprows=1)
# # # # # #     except Exception as e:
# # # # # #         st.error(f"Error reading CSV: {e}")
# # # # # #         st.stop()

# # # # # #     # --- Data Preprocessing ---
# # # # # #     st.header("Data Preprocessing")

# # # # # #     # Strip leading and trailing spaces from column names
# # # # # #     df.columns = df.columns.str.strip()

# # # # # #     # Drop columns with all missing values
# # # # # #     df.dropna(axis=1, how='all', inplace=True)

# # # # # #     # Fill missing values for categorical columns with 'Unknown'
# # # # # #     categorical_columns = df.select_dtypes(include=['object']).columns
# # # # # #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # # # #     # Convert date columns to datetime
# # # # # #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # # # # #     for col in date_columns:
# # # # # #         if col in df.columns:
# # # # # #             df[col] = pd.to_datetime(df[col], errors='coerce')

# # # # # #     # Fill missing values for numerical columns with the mean
# # # # # #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# # # # # #     # imputer = SimpleImputer(strategy='mean')
# # # # # #     # df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # # # #     # Encode categorical columns
# # # # # #     # label_encoders = {}
# # # # # #     # for col in categorical_columns:
# # # # # #     #     le = LabelEncoder()
# # # # # #     #     df[col] = le.fit_transform(df[col])
# # # # # #     #     label_encoders[col] = le

# # # # # #     # Scale numerical data
# # # # # #     # scaler = StandardScaler()
# # # # # #     # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # # # #     # --- Extract only the name from the contact person field
# # # # # #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0])

# # # # # #     # Create short forms for long names
# # # # # #     def create_short_form(name, max_length=10):
# # # # # #         if len(name) > max_length:
# # # # # #             return name[:max_length] + '...'
# # # # # #         return name

# # # # # #     # Apply short forms to specific columns
# # # # # #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # # # # #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # # # # #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # # # # #     df['Product Short'] = df['Product'].apply(create_short_form)
# # # # # #     # --- Visualizations ---

# # # # # #     # --- Visualizations Button ---
# # # # # #     if st.button("Show Visualizations"):
# # # # # #         st.header("Visualizations")

# # # # # #         # --- Helper function to display plots ---
# # # # # #         def display_plot(fig):
# # # # # #             st.pyplot(fig)
# # # # # #             plt.close(fig)  # Close the figure to prevent memory issues

# # # # # #         # --- Organize plots into columns ---
# # # # # #         col1, col2 = st.columns(2)

# # # # # #         with col1:
# # # # # #             # Opportunity Name Distribution
# # # # # #             fig_opportunity_hist, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             df['Opportunity Name Short'].value_counts().plot(kind='bar', ax=ax)
# # # # # #             ax.set_title('Opportunity Name Distribution', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Opportunity Name', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_opportunity_hist.tight_layout()
# # # # # #             display_plot(fig_opportunity_hist)

# # # # # #             # Contact Person Distribution
# # # # # #             fig_contact_hist, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             df['Contact Person Short'].value_counts().plot(kind='bar', ax=ax)
# # # # # #             ax.set_title('Contact Person Distribution', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Contact Person', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_contact_hist.tight_layout()
# # # # # #             display_plot(fig_contact_hist)

# # # # # #             # Contact Person Line Graph
# # # # # #             contact_person_counts = df['Contact Person Short'].value_counts().sort_index()
# # # # # #             fig_contact_line, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             sns.lineplot(x=contact_person_counts.index, y=contact_person_counts.values, ax=ax)
# # # # # #             ax.set_title('Contact Person Line Graph', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Contact Person', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_contact_line.tight_layout()
# # # # # #             display_plot(fig_contact_line)

# # # # # #         with col2:
# # # # # #             # Territory Name Distribution
# # # # # #             fig_territory_hist, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             df['Territory Name Short'].value_counts().plot(kind='bar', ax=ax)
# # # # # #             ax.set_title('Territory Name Distribution', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Territory Name', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_territory_hist.tight_layout()
# # # # # #             display_plot(fig_territory_hist)

# # # # # #             # Product Distribution
# # # # # #             fig_product_hist, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             df['Product Short'].value_counts().plot(kind='bar', ax=ax)
# # # # # #             ax.set_title('Product Distribution', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Product', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_product_hist.tight_layout()
# # # # # #             display_plot(fig_product_hist)

# # # # # #             # Product Line Graph
# # # # # #             product_counts = df['Product Short'].value_counts().sort_index()
# # # # # #             fig_product_line, ax = plt.subplots(figsize=(8, 5))  # Reduced figure size
# # # # # #             sns.lineplot(x=product_counts.index, y=product_counts.values, ax=ax)
# # # # # #             ax.set_title('Product Line Graph', fontsize=10)  # Smaller title
# # # # # #             ax.set_xlabel('Product', fontsize=8)
# # # # # #             ax.set_ylabel('Frequency', fontsize=8)
# # # # # #             plt.xticks(rotation=45, ha='right', fontsize=7)  # Rotate x-axis labels
# # # # # #             fig_product_line.tight_layout()
# # # # # #             display_plot(fig_product_line)

# # # # # #         # --- Additional plots below the columns ---
# # # # # #         st.subheader("Pie Chart Distributions")
# # # # # #         col3, col4 = st.columns(2)

# # # # # #         with col3:
# # # # # #             # Opportunity Name Pie Chart
# # # # # #             fig_opportunity_pie, ax = plt.subplots(figsize=(8, 5))
# # # # # #             df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, fontsize=8)
# # # # # #             ax.set_title('Opportunity Name Distribution', fontsize=10)
# # # # # #             ax.set_ylabel('')
# # # # # #             fig_opportunity_pie.tight_layout()
# # # # # #             display_plot(fig_opportunity_pie)

# # # # # #             # Contact Person Pie Chart
# # # # # #             fig_contact_pie, ax = plt.subplots(figsize=(8, 5))
# # # # # #             df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, fontsize=8)
# # # # # #             ax.set_title('Contact Person Distribution', fontsize=10)
# # # # # #             ax.set_ylabel('')
# # # # # #             fig_contact_pie.tight_layout()
# # # # # #             display_plot(fig_contact_pie)

# # # # # #         with col4:
# # # # # #             # Territory Name Pie Chart
# # # # # #             fig_territory_pie, ax = plt.subplots(figsize=(8, 5))
# # # # # #             df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, fontsize=8)
# # # # # #             ax.set_title('Territory Name Distribution', fontsize=10)
# # # # # #             ax.set_ylabel('')
# # # # # #             fig_territory_pie.tight_layout()
# # # # # #             display_plot(fig_territory_pie)

# # # # # #             # Product Pie Chart
# # # # # #             fig_product_pie, ax = plt.subplots(figsize=(8, 5))
# # # # # #             df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, fontsize=8)
# # # # # #             ax.set_title('Product Distribution', fontsize=10)
# # # # # #             ax.set_ylabel('')
# # # # # #             fig_product_pie.tight_layout()
# # # # # #             display_plot(fig_product_pie)

# # # # # # else:
# # # # # #     st.info("Please upload a CSV file to begin.")



# # # # # import os
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # # # from sklearn.impute import SimpleImputer
# # # # # from sklearn.ensemble import GradientBoostingRegressor
# # # # # from sklearn.metrics import mean_squared_error, r2_score
# # # # # import matplotlib.pyplot as plt
# # # # # import seaborn as sns
# # # # # import plotly.express as px

# # # # # # Create a directory to save images
# # # # # image_dir = 'images'
# # # # # if not os.path.exists(image_dir):
# # # # #     os.makedirs(image_dir)

# # # # # # Load the CSV file
# # # # # file_path = 'D:/skyward/Opportunity Details By Owner (27).csv'
# # # # # df = pd.read_csv(file_path, skiprows=1)  # Skip the first row which seems to be empty

# # # # # # Display the first few rows
# # # # # print(df.head())

# # # # # # Strip leading and trailing spaces from column names
# # # # # df.columns = df.columns.str.strip()

# # # # # # Check the actual column names
# # # # # print("Column names:", df.columns.tolist())

# # # # # # Data Preprocessing
# # # # # # Drop columns with all missing values
# # # # # df.dropna(axis=1, how='all', inplace=True)

# # # # # # Fill missing values for categorical columns with 'Unknown'
# # # # # categorical_columns = df.select_dtypes(include=['object']).columns
# # # # # df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # # # # Convert date columns to datetime
# # # # # date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # # # # for col in date_columns:
# # # # #     if col in df.columns:
# # # # #         df[col] = pd.to_datetime(df[col], errors='coerce')

# # # # # # Fill missing values for numerical columns with the mean
# # # # # numeric_columns = df.select_dtypes(include=[np.number]).columns
# # # # # imputer = SimpleImputer(strategy='mean')
# # # # # df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # # # # Encode categorical columns
# # # # # label_encoders = {}
# # # # # for col in categorical_columns:
# # # # #     le = LabelEncoder()
# # # # #     df[col] = le.fit_transform(df[col])
# # # # #     label_encoders[col] = le

# # # # # # Scale numerical data
# # # # # scaler = StandardScaler()
# # # # # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # # # # Feature Selection and Target Variable
# # # # # # For this example, let's predict 'Amount'
# # # # # target_column = 'Amount'
# # # # # features = df.drop(columns=[target_column])
# # # # # target = df[target_column]

# # # # # # Train-test split
# # # # # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # # # # Hyperparameter tuning using GridSearchCV
# # # # # param_grid = {
# # # # #     'n_estimators': [50, 100, 200],
# # # # #     'learning_rate': [0.01, 0.1, 0.2],
# # # # #     'max_depth': [3, 4, 5]
# # # # # }

# # # # # grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # # # #                            param_grid=param_grid,
# # # # #                            cv=5,
# # # # #                            scoring='r2',
# # # # #                            n_jobs=-1)

# # # # # grid_search.fit(X_train, y_train)

# # # # # # Best parameters and best score
# # # # # best_params = grid_search.best_params_
# # # # # best_score = grid_search.best_score_
# # # # # print(f"Best Parameters: {best_params}")
# # # # # print(f"Best Cross-Validation R^2 Score: {best_score}")

# # # # # # Train model with best parameters
# # # # # best_model = grid_search.best_estimator_
# # # # # best_model.fit(X_train, y_train)

# # # # # # Make predictions with the best model
# # # # # y_pred_best = best_model.predict(X_test)

# # # # # # Evaluate the best model
# # # # # mse_best = mean_squared_error(y_test, y_pred_best)
# # # # # r2_best = r2_score(y_test, y_pred_best)
# # # # # print(f"Mean Squared Error (Best Model): {mse_best}")
# # # # # print(f"R^2 Score (Best Model): {r2_best}")

# # # # # # Provide insights for new customers
# # # # # feature_importances = best_model.feature_importances_
# # # # # features = X_train.columns
# # # # # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # # # # importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # # # # print("\nTop 5 Features that influence the Amount:")
# # # # # top_5_features = importance_df.head(5)['Feature'].tolist()
# # # # # print(importance_df.head(10))

# # # # # # Decode categorical features back to their original labels
# # # # # for col in categorical_columns:
# # # # #     df[col] = label_encoders[col].inverse_transform(df[col])

# # # # # # Extract only the name from the contact person field
# # # # # df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0])

# # # # # # Create short forms for long names
# # # # # def create_short_form(name, max_length=4):
# # # # #     if len(name) > max_length:
# # # # #         return name[:max_length] + '...'
# # # # #     return name

# # # # # # Create short forms for product names by taking only the first word
# # # # # df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0])
# # # # # def create_product_short_form(name):
# # # # #     return name.split()[0]

# # # # # # Apply short forms to specific columns
# # # # # df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # # # # df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # # # # df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # # # # df['Product Short'] = df['Product'].apply(create_product_short_form)

# # # # # # Plot histograms for Opportunity Name, Territory Name, and Top 5 Features using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Opportunity Name Short'], kde=True)
# # # # # plt.title('Opportunity Name Distribution')
# # # # # plt.xlabel('Opportunity Name')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_distribution_histogram.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Territory Name Short'], kde=True)
# # # # # plt.title('Territory Name Distribution')
# # # # # plt.xlabel('Territory Name')
# # # # # plt.ylabel('Frequency')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_distribution_histogram.png'))
# # # # # plt.close()

# # # # # for feature in top_5_features:
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     sns.histplot(df[feature], kde=True)
# # # # #     plt.title(f'{feature} Distribution')
# # # # #     plt.xlabel(feature)
# # # # #     plt.ylabel('Frequency')
# # # # #     plt.tight_layout()
# # # # #     plt.savefig(os.path.join(image_dir, f'{feature}_distribution_histogram.png'))
# # # # #     plt.close()

# # # # # # Plot line graphs for Product and Top 5 Features using short forms
# # # # # product_counts = df['Product Short'].value_counts().sort_index()
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.lineplot(x=product_counts.index, y=product_counts.values)
# # # # # plt.title('Product Line Graph')
# # # # # plt.xlabel('Product')
# # # # # plt.ylabel('Frequency')
# # # # # plt.xticks(rotation=45, ha='right')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_line_graph.png'))
# # # # # plt.close()

# # # # # for feature in top_5_features:
# # # # #     if feature not in ['Close Date', 'Modified Date']:
# # # # #         feature_counts = df[feature].value_counts().sort_index()
# # # # #         plt.figure(figsize=(10, 6))
# # # # #         sns.lineplot(x=feature_counts.index, y=feature_counts.values)
# # # # #         plt.title(f'{feature} Line Graph')
# # # # #         plt.xlabel(feature)
# # # # #         plt.ylabel('Frequency')
# # # # #         plt.xticks(rotation=45, ha='right')
# # # # #         plt.tight_layout()
# # # # #         plt.savefig(os.path.join(image_dir, f'{feature}_line_graph.png'))
# # # # #         plt.close()

# # # # # # Plot density plots for Opportunity Name, Territory Name, and Top 5 Features using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Opportunity Name Short'], kde=True)
# # # # # plt.title('Opportunity Name Density')
# # # # # plt.xlabel('Opportunity Name')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_density.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # sns.histplot(df['Territory Name Short'], kde=True)
# # # # # plt.title('Territory Name Density')
# # # # # plt.xlabel('Territory Name')
# # # # # plt.ylabel('Density')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_density.png'))
# # # # # plt.close()

# # # # # for feature in top_5_features:
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     sns.histplot(df[feature], kde=True)
# # # # #     plt.title(f'{feature} Density')
# # # # #     plt.xlabel(feature)
# # # # #     plt.ylabel('Density')
# # # # #     plt.tight_layout()
# # # # #     plt.savefig(os.path.join(image_dir, f'{feature}_density.png'))
# # # # #     plt.close()

# # # # # # Plot pie charts for Opportunity Name, Territory Name, Product, and Top 5 Features using short forms
# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Opportunity Name Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'opportunity_name_distribution_pie.png'))
# # # # # plt.close()


# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Contact Person Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'contact_person_distribution_pie.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Territory Name Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'territory_name_distribution_pie.png'))
# # # # # plt.close()

# # # # # plt.figure(figsize=(10, 6))
# # # # # df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # plt.title('Product Distribution')
# # # # # plt.ylabel('')
# # # # # plt.tight_layout()
# # # # # plt.savefig(os.path.join(image_dir, 'product_distribution_pie.png'))
# # # # # plt.close()






# # # # # import streamlit as st
# # # # # import os
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # # # from sklearn.impute import SimpleImputer
# # # # # from sklearn.ensemble import GradientBoostingRegressor
# # # # # from sklearn.metrics import mean_squared_error, r2_score
# # # # # import matplotlib.pyplot as plt
# # # # # import seaborn as sns

# # # # # # Load the CSV file
# # # # # file_path = 'D:/skyward/Opportunity Details By Owner (27).csv'
# # # # # try:
# # # # #     df = pd.read_csv(file_path, skiprows=1)  # Skip the first row which seems to be empty
# # # # # except FileNotFoundError:
# # # # #     st.error(f"Error: The file {file_path} was not found. Please check the file path.")
# # # # #     st.stop()

# # # # # # --- Data Preprocessing ---
# # # # # # Strip leading and trailing spaces from column names
# # # # # df.columns = df.columns.str.strip()

# # # # # # Drop columns with all missing values
# # # # # df.dropna(axis=1, how='all', inplace=True)

# # # # # # Fill missing values for categorical columns with 'Unknown'
# # # # # categorical_columns = df.select_dtypes(include=['object']).columns
# # # # # df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # # # # Convert date columns to datetime
# # # # # date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # # # # for col in date_columns:
# # # # #     if col in df.columns:
# # # # #         df[col] = pd.to_datetime(df[col], errors='coerce')

# # # # # # Fill missing values for numerical columns with the mean
# # # # # numeric_columns = df.select_dtypes(include=[np.number]).columns
# # # # # imputer = SimpleImputer(strategy='mean')
# # # # # df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # # # # Encode categorical columns
# # # # # label_encoders = {}
# # # # # for col in categorical_columns:
# # # # #     le = LabelEncoder()
# # # # #     df[col] = le.fit_transform(df[col])
# # # # #     label_encoders[col] = le

# # # # # # Scale numerical data
# # # # # scaler = StandardScaler()
# # # # # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # # # # --- Feature Selection and Model Training ---
# # # # # # Target variable
# # # # # target_column = 'Amount'
# # # # # features = df.drop(columns=[target_column])
# # # # # target = df[target_column]

# # # # # # Train-test split
# # # # # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # # # # Hyperparameter tuning using GridSearchCV
# # # # # param_grid = {
# # # # #     'n_estimators': [50, 100, 200],
# # # # #     'learning_rate': [0.01, 0.1, 0.2],
# # # # #     'max_depth': [3, 4, 5]
# # # # # }

# # # # # grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # # # #                            param_grid=param_grid,
# # # # #                            cv=5,
# # # # #                            scoring='r2',
# # # # #                            n_jobs=-1)

# # # # # grid_search.fit(X_train, y_train)

# # # # # # Best parameters and best score
# # # # # best_params = grid_search.best_params_
# # # # # best_score = grid_search.best_score_

# # # # # # Train model with best parameters
# # # # # best_model = grid_search.best_estimator_
# # # # # best_model.fit(X_train, y_train)

# # # # # # Evaluate the best model
# # # # # y_pred_best = best_model.predict(X_test)
# # # # # mse_best = mean_squared_error(y_test, y_pred_best)
# # # # # r2_best = r2_score(y_test, y_pred_best)

# # # # # # Feature Importance
# # # # # feature_importances = best_model.feature_importances_
# # # # # features = X_train.columns
# # # # # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # # # # importance_df = importance_df.sort_values(by='Importance', ascending=False)
# # # # # top_5_features = importance_df.head(5)['Feature'].tolist()

# # # # # # Decode categorical features back to original labels
# # # # # for col in categorical_columns:
# # # # #     df[col] = label_encoders[col].inverse_transform(df[col])

# # # # # # --- Short Form Functions ---
# # # # # def create_short_form(name, max_length=4):
# # # # #     if isinstance(name, str):  # Check if the value is a string
# # # # #         if len(name) > max_length:
# # # # #             return name[:max_length] + '...'
# # # # #         return name
# # # # #     return name

# # # # # def create_product_short_form(name):
# # # # #     if isinstance(name, str):  # Check if the value is a string
# # # # #         return name.split()[0]
# # # # #     return name

# # # # # # --- Applying Short Forms ---
# # # # # # Extract only the name from the contact person field
# # # # # df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else x)

# # # # # df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # # # # df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # # # # df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # # # # df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else x)
# # # # # df['Product Short'] = df['Product'].apply(create_product_short_form)

# # # # # # --- Streamlit App ---
# # # # # st.title('Opportunity Analysis Dashboard')


# # # # # # --- Plots ---
# # # # # st.header('Data Visualizations')

# # # # # # --- Histograms ---
# # # # # st.subheader('Histograms')

# # # # # fig1, ax1 = plt.subplots(figsize=(10, 6))
# # # # # sns.histplot(df['Opportunity Name Short'], kde=True, ax=ax1)
# # # # # ax1.set_title('Opportunity Name Distribution')
# # # # # ax1.set_xlabel('Opportunity Name')
# # # # # ax1.set_ylabel('Frequency')
# # # # # st.pyplot(fig1)

# # # # # fig2, ax2 = plt.subplots(figsize=(10, 6))
# # # # # sns.histplot(df['Territory Name Short'], kde=True, ax=ax2)
# # # # # ax2.set_title('Territory Name Distribution')
# # # # # ax2.set_xlabel('Territory Name')
# # # # # ax2.set_ylabel('Frequency')
# # # # # st.pyplot(fig2)

# # # # # for feature in top_5_features:
# # # # #     fig3, ax3 = plt.subplots(figsize=(10, 6))
# # # # #     sns.histplot(df[feature], kde=True, ax=ax3)
# # # # #     ax3.set_title(f'{feature} Distribution')
# # # # #     ax3.set_xlabel(feature)
# # # # #     ax3.set_ylabel('Frequency')
# # # # #     st.pyplot(fig3)

# # # # # # --- Line Graphs ---
# # # # # st.subheader('Line Graphs')

# # # # # product_counts = df['Product Short'].value_counts().sort_index()
# # # # # fig4, ax4 = plt.subplots(figsize=(10, 6))
# # # # # sns.lineplot(x=product_counts.index, y=product_counts.values, ax=ax4)
# # # # # ax4.set_title('Product Line Graph')
# # # # # ax4.set_xlabel('Product')
# # # # # ax4.set_ylabel('Frequency')
# # # # # ax4.tick_params(axis='x', rotation=45)
# # # # # st.pyplot(fig4)

# # # # # for feature in top_5_features:
# # # # #     if feature not in ['Close Date', 'Modified Date']:
# # # # #         feature_counts = df[feature].value_counts().sort_index()
# # # # #         fig5, ax5 = plt.subplots(figsize=(10, 6))
# # # # #         sns.lineplot(x=feature_counts.index, y=feature_counts.values, ax=ax5)
# # # # #         ax5.set_title(f'{feature} Line Graph')
# # # # #         ax5.set_xlabel(feature)
# # # # #         ax5.set_ylabel('Frequency')
# # # # #         ax5.tick_params(axis='x', rotation=45)
# # # # #         st.pyplot(fig5)

# # # # # # --- Density Plots ---
# # # # # st.subheader('Density Plots')

# # # # # #Remove the graphs for ['Opportunity Name Short'] and ['Territory Name Short']

# # # # # for feature in top_5_features:
# # # # #     try:
# # # # #         fig8, ax8 = plt.subplots(figsize=(10, 6))
# # # # #         sns.kdeplot(df[feature], fill=True, ax=ax8)
# # # # #         ax8.set_title(f'{feature} Density')
# # # # #         ax8.set_xlabel(feature)
# # # # #         ax8.set_ylabel('Density')
# # # # #         st.pyplot(fig8)
# # # # #     except TypeError as e:
# # # # #         st.warning(f"Skipping density plot for {feature} due to TypeError: {e}")

# # # # # # --- Pie Charts ---
# # # # # st.subheader('Pie Charts')

# # # # # fig9, ax9 = plt.subplots(figsize=(10, 6))
# # # # # df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # ax9.set_title('Opportunity Name Distribution')
# # # # # ax9.set_ylabel('')
# # # # # st.pyplot(fig9)

# # # # # fig10, ax10 = plt.subplots(figsize=(10, 6))
# # # # # df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # ax10.set_title('Contact Person Distribution')
# # # # # ax10.set_ylabel('')
# # # # # st.pyplot(fig10)

# # # # # fig11, ax11 = plt.subplots(figsize=(10, 6))
# # # # # df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # ax11.set_title('Territory Name Distribution')
# # # # # ax11.set_ylabel('')
# # # # # st.pyplot(fig11)

# # # # # fig12, ax12 = plt.subplots(figsize=(10, 6))
# # # # # df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # # # ax12.set_title('Product Distribution')
# # # # # ax12.set_ylabel('')
# # # # # st.pyplot(fig12)





# # # # import streamlit as st
# # # # import os
# # # # import pandas as pd
# # # # import numpy as np
# # # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # # from sklearn.impute import SimpleImputer
# # # # from sklearn.ensemble import GradientBoostingRegressor
# # # # from sklearn.metrics import mean_squared_error, r2_score
# # # # import matplotlib.pyplot as plt
# # # # import seaborn as sns

# # # # # --- Helper Functions ---
# # # # def create_short_form(name, max_length=4):
# # # #     if isinstance(name, str):
# # # #         if len(name) > max_length:
# # # #             return name[:max_length] + '...'
# # # #         return name
# # # #     return str(name)  # Handle non-string values

# # # # def create_product_short_form(name):
# # # #     if isinstance(name, str):
# # # #         return name.split()[0]
# # # #     return str(name)  # Handle non-string values

# # # # # --- Streamlit App ---
# # # # st.title('Opportunity Analysis Dashboard')

# # # # # File Upload
# # # # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # # # if uploaded_file is not None:
# # # #     # Load the CSV file
# # # #     try:
# # # #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row which seems to be empty
# # # #     except Exception as e:
# # # #         st.error(f"Error loading CSV: {e}")
# # # #         st.stop()

# # # #     # --- Data Preprocessing ---
# # # #     # Strip leading and trailing spaces from column names
# # # #     df.columns = df.columns.str.strip()

# # # #     # Drop columns with all missing values
# # # #     df.dropna(axis=1, how='all', inplace=True)

# # # #     # Fill missing values for categorical columns with 'Unknown'
# # # #     categorical_columns = df.select_dtypes(include=['object']).columns
# # # #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # #     # Convert date columns to datetime
# # # #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # # #     for col in date_columns:
# # # #         if col in df.columns:
# # # #             df[col] = pd.to_datetime(df[col], errors='coerce')

# # # #     # Fill missing values for numerical columns with the mean
# # # #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# # # #     imputer = SimpleImputer(strategy='mean')
# # # #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # #     # Encode categorical columns
# # # #     label_encoders = {}
# # # #     for col in categorical_columns:
# # # #         le = LabelEncoder()
# # # #         df[col] = le.fit_transform(df[col])
# # # #         label_encoders[col] = le

# # # #     # Scale numerical data
# # # #     scaler = StandardScaler()
# # # #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # #     # --- Feature Selection and Model Training ---
# # # #     # Target variable
# # # #     target_column = 'Amount'
# # # #     features = df.drop(columns=[target_column])
# # # #     target = df[target_column]

# # # #     # Train-test split
# # # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # #     # Hyperparameter tuning using GridSearchCV
# # # #     param_grid = {
# # # #         'n_estimators': [50, 100, 200],
# # # #         'learning_rate': [0.01, 0.1, 0.2],
# # # #         'max_depth': [3, 4, 5]
# # # #     }

# # # #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # # #                                param_grid=param_grid,
# # # #                                cv=5,
# # # #                                scoring='r2',
# # # #                                n_jobs=-1)

# # # #     grid_search.fit(X_train, y_train)

# # # #     # Best parameters and best score
# # # #     best_params = grid_search.best_params_
# # # #     best_score = grid_search.best_score_

# # # #     # Train model with best parameters
# # # #     best_model = grid_search.best_estimator_
# # # #     best_model.fit(X_train, y_train)

# # # #     # Evaluate the best model
# # # #     y_pred_best = best_model.predict(X_test)
# # # #     mse_best = mean_squared_error(y_test, y_pred_best)
# # # #     r2_best = r2_score(y_test, y_pred_best)

# # # #     # Feature Importance
# # # #     feature_importances = best_model.feature_importances_
# # # #     features = X_train.columns
# # # #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # # #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# # # #     top_5_features = importance_df.head(5)['Feature'].tolist()

# # # #     # Decode categorical features back to original labels
# # # #     for col in categorical_columns:
# # # #         df[col] = label_encoders[col].inverse_transform(df[col])

# # # #     # --- Applying Short Forms ---
# # # #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # # #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # # #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # # #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # # #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # # #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# # # #     # --- Display Model Performance and Feature Importance ---
# # # #     st.header('Model Performance')
# # # #     st.write(f"Best Parameters: {best_params}")
# # # #     st.write(f"Best Cross-Validation R^2 Score: {best_score}")
# # # #     st.write(f"Mean Squared Error (Best Model): {mse_best}")
# # # #     st.write(f"R^2 Score (Best Model): {r2_best}")

# # # #     st.header('Feature Importance')
# # # #     st.dataframe(importance_df.head(10))

# # # #     # --- Visualization Buttons ---
# # # #     if st.button('Show Histograms'):
# # # #         st.header('Histograms')

# # # #         fig1, ax1 = plt.subplots(figsize=(10, 6))
# # # #         sns.histplot(df['Opportunity Name Short'], kde=True, ax=ax1)
# # # #         ax1.set_title('Opportunity Name Distribution')
# # # #         ax1.set_xlabel('Opportunity Name')
# # # #         ax1.set_ylabel('Frequency')
# # # #         st.pyplot(fig1)

# # # #         fig2, ax2 = plt.subplots(figsize=(10, 6))
# # # #         sns.histplot(df['Territory Name Short'], kde=True, ax=ax2)
# # # #         ax2.set_title('Territory Name Distribution')
# # # #         ax2.set_xlabel('Territory Name')
# # # #         ax2.set_ylabel('Frequency')
# # # #         st.pyplot(fig2)

# # # #         for feature in top_5_features:
# # # #             fig3, ax3 = plt.subplots(figsize=(10, 6))
# # # #             sns.histplot(df[feature], kde=True, ax=ax3)
# # # #             ax3.set_title(f'{feature} Distribution')
# # # #             ax3.set_xlabel(feature)
# # # #             ax3.set_ylabel('Frequency')
# # # #             st.pyplot(fig3)

# # # #     if st.button('Show Line Graphs'):
# # # #         st.header('Line Graphs')

# # # #         product_counts = df['Product Short'].value_counts().sort_index()
# # # #         fig4, ax4 = plt.subplots(figsize=(10, 6))
# # # #         sns.lineplot(x=product_counts.index, y=product_counts.values, ax=ax4)
# # # #         ax4.set_title('Product Line Graph')
# # # #         ax4.set_xlabel('Product')
# # # #         ax4.set_ylabel('Frequency')
# # # #         ax4.tick_params(axis='x', rotation=45)
# # # #         st.pyplot(fig4)

# # # #         for feature in top_5_features:
# # # #             if feature not in ['Close Date', 'Modified Date']:
# # # #                 feature_counts = df[feature].value_counts().sort_index()
# # # #                 fig5, ax5 = plt.subplots(figsize=(10, 6))
# # # #                 sns.lineplot(x=feature_counts.index, y=feature_counts.values, ax=ax5)
# # # #                 ax5.set_title(f'{feature} Line Graph')
# # # #                 ax5.set_xlabel(feature)
# # # #                 ax5.set_ylabel('Frequency')
# # # #                 ax5.tick_params(axis='x', rotation=45)
# # # #                 st.pyplot(fig5)

# # # #     if st.button('Show Density Plots'):
# # # #         st.header('Density Plots')
# # # #         #Remove the graphs for ['Opportunity Name Short'] and ['Territory Name Short']

# # # #         for feature in top_5_features:
# # # #             try:
# # # #                 fig8, ax8 = plt.subplots(figsize=(10, 6))
# # # #                 sns.kdeplot(df[feature], fill=True, ax=ax8)
# # # #                 ax8.set_title(f'{feature} Density')
# # # #                 ax8.set_xlabel(feature)
# # # #                 ax8.set_ylabel('Density')
# # # #                 st.pyplot(fig8)
# # # #             except TypeError as e:
# # # #                 st.warning(f"Skipping density plot for {feature} due to TypeError: {e}")

# # # #     if st.button('Show Pie Charts'):
# # # #         st.header('Pie Charts')

# # # #         fig9, ax9 = plt.subplots(figsize=(10, 6))
# # # #         df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # #         ax9.set_title('Opportunity Name Distribution')
# # # #         ax9.set_ylabel('')
# # # #         st.pyplot(fig9)

# # # #         fig10, ax10 = plt.subplots(figsize=(10, 6))
# # # #         df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # #         ax10.set_title('Contact Person Distribution')
# # # #         ax10.set_ylabel('')
# # # #         st.pyplot(fig10)

# # # #         fig11, ax11 = plt.subplots(figsize=(10, 6))
# # # #         df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # #         ax11.set_title('Territory Name Distribution')
# # # #         ax11.set_ylabel('')
# # # #         st.pyplot(fig11)

# # # #         fig12, ax12 = plt.subplots(figsize=(10, 6))
# # # #         df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # # #         ax12.set_title('Product Distribution')
# # # #         ax12.set_ylabel('')
# # # #         st.pyplot(fig12)






# # # import streamlit as st
# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.ensemble import GradientBoostingRegressor
# # # from sklearn.metrics import mean_squared_error, r2_score
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # # --- Helper Functions ---
# # # def create_short_form(name, max_length=4):
# # #     if isinstance(name, str):
# # #         if len(name) > max_length:
# # #             return name[:max_length] + '...'
# # #         return name
# # #     return str(name)  # Handle non-string values

# # # def create_product_short_form(name):
# # #     if isinstance(name, str):
# # #         return name.split()[0]
# # #     return str(name)  # Handle non-string values

# # # # --- Streamlit App ---
# # # st.title('Opportunity Analysis Dashboard')

# # # # File Upload
# # # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # # if uploaded_file is not None:
# # #     # Load the CSV file
# # #     try:
# # #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row which seems to be empty
# # #     except Exception as e:
# # #         st.error(f"Error loading CSV: {e}")
# # #         st.stop()

# # #     # --- Data Preprocessing ---
# # #     # Strip leading and trailing spaces from column names
# # #     df.columns = df.columns.str.strip()

# # #     # Drop columns with all missing values
# # #     df.dropna(axis=1, how='all', inplace=True)

# # #     # Fill missing values for categorical columns with 'Unknown'
# # #     categorical_columns = df.select_dtypes(include=['object']).columns
# # #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # #     # Convert date columns to datetime
# # #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # #     for col in date_columns:
# # #         if col in df.columns:
# # #             df[col] = pd.to_datetime(df[col], errors='coerce')

# # #     # Fill missing values for numerical columns with the mean
# # #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# # #     imputer = SimpleImputer(strategy='mean')
# # #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # #     # Encode categorical columns
# # #     label_encoders = {}
# # #     for col in categorical_columns:
# # #         le = LabelEncoder()
# # #         df[col] = le.fit_transform(df[col])
# # #         label_encoders[col] = le

# # #     # Scale numerical data
# # #     scaler = StandardScaler()
# # #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # #     # --- Feature Selection and Model Training ---
# # #     # Target variable
# # #     target_column = 'Amount'
# # #     features = df.drop(columns=[target_column])
# # #     target = df[target_column]

# # #     # Train-test split
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # #     # Hyperparameter tuning using GridSearchCV
# # #     param_grid = {
# # #         'n_estimators': [50, 100, 200],
# # #         'learning_rate': [0.01, 0.1, 0.2],
# # #         'max_depth': [3, 4, 5]
# # #     }

# # #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # #                                param_grid=param_grid,
# # #                                cv=5,
# # #                                scoring='r2',
# # #                                n_jobs=-1)

# # #     grid_search.fit(X_train, y_train)

# # #     # Best parameters and best score
# # #     best_params = grid_search.best_params_
# # #     best_score = grid_search.best_score_

# # #     # Train model with best parameters
# # #     best_model = grid_search.best_estimator_
# # #     best_model.fit(X_train, y_train)

# # #     # Evaluate the best model
# # #     y_pred_best = best_model.predict(X_test)
# # #     mse_best = mean_squared_error(y_test, y_pred_best)
# # #     r2_best = r2_score(y_test, y_pred_best)

# # #     # Feature Importance
# # #     feature_importances = best_model.feature_importances_
# # #     features = X_train.columns
# # #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# # #     top_5_features = importance_df.head(5)['Feature'].tolist()

# # #     # Decode categorical features back to original labels
# # #     for col in categorical_columns:
# # #         df[col] = label_encoders[col].inverse_transform(df[col])

# # #     # --- Applying Short Forms ---
# # #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# # #     # --- Display Model Performance and Feature Importance ---
# # #     st.header('Model Performance')
# # #     st.write(f"Best Parameters: {best_params}")
# # #     st.write(f"Best Cross-Validation R^2 Score: {best_score}")
# # #     st.write(f"Mean Squared Error (Best Model): {mse_best}")
# # #     st.write(f"R^2 Score (Best Model): {r2_best}")

# # #     st.header('Feature Importance')
# # #     st.dataframe(importance_df.head(10))

# # #     # --- Visualization Buttons ---
# # #     graph_type = st.selectbox("Select Graph Type", ["Histograms", "Line Graphs", "Density Plots", "Pie Charts"])

# # #     if graph_type == "Histograms":
# # #         st.header('Histograms')

# # #         fig1, ax1 = plt.subplots(figsize=(10, 6))
# # #         sns.histplot(df['Opportunity Name Short'], kde=True, ax=ax1)
# # #         ax1.set_title('Opportunity Name Distribution')
# # #         ax1.set_xlabel('Opportunity Name')
# # #         ax1.set_ylabel('Frequency')
# # #         st.pyplot(fig1)

# # #         fig2, ax2 = plt.subplots(figsize=(10, 6))
# # #         sns.histplot(df['Territory Name Short'], kde=True, ax=ax2)
# # #         ax2.set_title('Territory Name Distribution')
# # #         ax2.set_xlabel('Territory Name')
# # #         ax2.set_ylabel('Frequency')
# # #         st.pyplot(fig2)

# # #         for feature in top_5_features:
# # #             fig3, ax3 = plt.subplots(figsize=(10, 6))
# # #             sns.histplot(df[feature], kde=True, ax=ax3)
# # #             ax3.set_title(f'{feature} Distribution')
# # #             ax3.set_xlabel(feature)
# # #             ax3.set_ylabel('Frequency')
# # #             st.pyplot(fig3)

# # #     elif graph_type == "Line Graphs":
# # #         st.header('Line Graphs')

# # #         product_counts = df['Product Short'].value_counts().sort_index()
# # #         fig4, ax4 = plt.subplots(figsize=(10, 6))
# # #         sns.lineplot(x=product_counts.index, y=product_counts.values, ax=ax4)
# # #         ax4.set_title('Product Line Graph')
# # #         ax4.set_xlabel('Product')
# # #         ax4.set_ylabel('Frequency')
# # #         ax4.tick_params(axis='x', rotation=45)
# # #         st.pyplot(fig4)

# # #         for feature in top_5_features:
# # #             if feature not in ['Close Date', 'Modified Date']:
# # #                 feature_counts = df[feature].value_counts().sort_index()
# # #                 fig5, ax5 = plt.subplots(figsize=(10, 6))
# # #                 sns.lineplot(x=feature_counts.index, y=feature_counts.values, ax=ax5)
# # #                 ax5.set_title(f'{feature} Line Graph')
# # #                 ax5.set_xlabel(feature)
# # #                 ax5.set_ylabel('Frequency')
# # #                 ax5.tick_params(axis='x', rotation=45)
# # #                 st.pyplot(fig5)

# # #     elif graph_type == "Density Plots":
# # #         st.header('Density Plots')

# # #         for feature in top_5_features:
# # #             try:
# # #                 fig8, ax8 = plt.subplots(figsize=(10, 6))
# # #                 sns.kdeplot(df[feature], fill=True, ax=ax8)
# # #                 ax8.set_title(f'{feature} Density')
# # #                 ax8.set_xlabel(feature)
# # #                 ax8.set_ylabel('Density')
# # #                 st.pyplot(fig8)
# # #             except TypeError as e:
# # #                 st.warning(f"Skipping density plot for {feature} due to TypeError: {e}")

# # #     elif graph_type == "Pie Charts":
# # #         st.header('Pie Charts')

# # #         fig9, ax9 = plt.subplots(figsize=(10, 6))
# # #         df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax9.set_title('Opportunity Name Distribution')
# # #         ax9.set_ylabel('')
# # #         st.pyplot(fig9)

# # #         fig10, ax10 = plt.subplots(figsize=(10, 6))
# # #         df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax10.set_title('Contact Person Distribution')
# # #         ax10.set_ylabel('')
# # #         st.pyplot(fig10)

# # #         fig11, ax11 = plt.subplots(figsize=(10, 6))
# # #         df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax11.set_title('Territory Name Distribution')
# # #         ax11.set_ylabel('')
# # #         st.pyplot(fig11)

# # #         fig12, ax12 = plt.subplots(figsize=(10, 6))
# # #         df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax12.set_title('Product Distribution')
# # #         ax12.set_ylabel('')
# # #         st.pyplot(fig12)




# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.ensemble import GradientBoostingRegressor
# # # from sklearn.metrics import mean_squared_error, r2_score
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # # --- Helper Functions ---
# # # def create_short_form(name, max_length=4):
# # #     if isinstance(name, str):
# # #         if len(name) > max_length:
# # #             return name[:max_length] + '...'
# # #         return name
# # #     return str(name)  # Handle non-string values

# # # def create_product_short_form(name):
# # #     if isinstance(name, str):
# # #         return name.split()[0]
# # #     return str(name)  # Handle non-string values

# # # # --- Streamlit App ---
# # # st.set_page_config(layout="wide")  # Use the full width of the page

# # # # Add a title and a brief description
# # # st.title('Opportunity Analysis Dashboard')
# # # st.markdown("Analyze your opportunity data and gain insights with this interactive dashboard.")

# # # # File Upload
# # # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # # if uploaded_file is not None:
# # #     # Load the CSV file
# # #     try:
# # #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row
# # #     except Exception as e:
# # #         st.error(f"Error loading CSV: {e}")
# # #         st.stop()

# # #     # --- Data Preprocessing ---
# # #     df.columns = df.columns.str.strip()
# # #     df.dropna(axis=1, how='all', inplace=True)
# # #     categorical_columns = df.select_dtypes(include=['object']).columns
# # #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')
# # #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # #     for col in date_columns:
# # #         if col in df.columns:
# # #             df[col] = pd.to_datetime(df[col], errors='coerce')
# # #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# # #     imputer = SimpleImputer(strategy='mean')
# # #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
# # #     label_encoders = {}
# # #     for col in categorical_columns:
# # #         le = LabelEncoder()
# # #         df[col] = le.fit_transform(df[col])
# # #         label_encoders[col] = le
# # #     scaler = StandardScaler()
# # #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # #     # --- Feature Selection and Model Training ---
# # #     target_column = 'Amount'
# # #     features = df.drop(columns=[target_column])
# # #     target = df[target_column]
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #     param_grid = {
# # #         'n_estimators': [50, 100, 200],
# # #         'learning_rate': [0.01, 0.1, 0.2],
# # #         'max_depth': [3, 4, 5]
# # #     }
# # #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # #                                param_grid=param_grid,
# # #                                cv=5,
# # #                                scoring='r2',
# # #                                n_jobs=-1)
# # #     grid_search.fit(X_train, y_train)
# # #     best_params = grid_search.best_params_
# # #     best_score = grid_search.best_score_
# # #     best_model = grid_search.best_estimator_
# # #     best_model.fit(X_train, y_train)
# # #     y_pred_best = best_model.predict(X_test)
# # #     mse_best = mean_squared_error(y_test, y_pred_best)
# # #     r2_best = r2_score(y_test, y_pred_best)
# # #     feature_importances = best_model.feature_importances_
# # #     features = X_train.columns
# # #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# # #     top_5_features = importance_df.head(5)['Feature'].tolist()
# # #     for col in categorical_columns:
# # #         df[col] = label_encoders[col].inverse_transform(df[col])

# # #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# # #     # --- Layout with Columns ---
# # #     #col1, col2 = st.columns([1, 2])  # Adjust column widths as needed

# # #     #with col1:
# # #         #st.header("Model Performance")
# # #         #st.write(f"**Best Parameters:** {best_params}")
# # #         #st.write(f"**Cross-Validation R^2:** {best_score:.4f}")
# # #         #st.write(f"**Mean Squared Error:** {mse_best:.2f}")
# # #         #st.write(f"**R^2 Score:** {r2_best:.4f}")

# # #         #st.header("Top Features")
# # #         #st.dataframe(importance_df.head(5), height=250)

# # #     #with col2:
# # #     st.header("Visualizations")
# # #     graph_type = st.selectbox("Select Graph Type", ["Histograms", "Line Graphs", "Pie Charts"]) #Removed Density plots due to errors

# # #     if graph_type == "Histograms":
# # #         st.subheader('Histograms')

# # #         fig1, ax1 = plt.subplots(figsize=(8, 4))
# # #         sns.histplot(df['Opportunity Name Short'], kde=True, ax=ax1)
# # #         ax1.set_title('Opportunity Name Distribution')
# # #         ax1.set_xlabel('Opportunity Name')
# # #         ax1.set_ylabel('Frequency')
# # #         st.pyplot(fig1)

# # #         fig2, ax2 = plt.subplots(figsize=(8, 4))
# # #         sns.histplot(df['Territory Name Short'], kde=True, ax=ax2)
# # #         ax2.set_title('Territory Name Distribution')
# # #         ax2.set_xlabel('Territory Name')
# # #         ax2.set_ylabel('Frequency')
# # #         st.pyplot(fig2)

# # #         for feature in top_5_features:
# # #             fig3, ax3 = plt.subplots(figsize=(8, 4))
# # #             sns.histplot(df[feature], kde=True, ax=ax3)
# # #             ax3.set_title(f'{feature} Distribution')
# # #             ax3.set_xlabel(feature)
# # #             ax3.set_ylabel('Frequency')
# # #             st.pyplot(fig3)

# # #     elif graph_type == "Line Graphs":
# # #         st.subheader('Line Graphs')

# # #         product_counts = df['Product Short'].value_counts().sort_index()
# # #         fig4, ax4 = plt.subplots(figsize=(8, 4))
# # #         sns.lineplot(x=product_counts.index, y=product_counts.values, ax=ax4)
# # #         ax4.set_title('Product Line Graph')
# # #         ax4.set_xlabel('Product')
# # #         ax4.set_ylabel('Frequency')
# # #         ax4.tick_params(axis='x', rotation=45)
# # #         st.pyplot(fig4)

# # #         for feature in top_5_features:
# # #             if feature not in ['Close Date', 'Modified Date']:
# # #                 feature_counts = df[feature].value_counts().sort_index()
# # #                 fig5, ax5 = plt.subplots(figsize=(8, 4))
# # #                 sns.lineplot(x=feature_counts.index, y=feature_counts.values, ax=ax5)
# # #                 ax5.set_title(f'{feature} Line Graph')
# # #                 ax5.set_xlabel(feature)
# # #                 ax5.set_ylabel('Frequency')
# # #                 ax5.tick_params(axis='x', rotation=45)
# # #                 st.pyplot(fig5)

# # #     elif graph_type == "Pie Charts":
# # #         st.subheader('Pie Charts')

# # #         fig9, ax9 = plt.subplots(figsize=(8, 4))
# # #         df['Opportunity Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax9.set_title('Opportunity Name Distribution')
# # #         ax9.set_ylabel('')
# # #         st.pyplot(fig9)

# # #         fig10, ax10 = plt.subplots(figsize=(8, 4))
# # #         df['Contact Person Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax10.set_title('Contact Person Distribution')
# # #         ax10.set_ylabel('')
# # #         st.pyplot(fig10)

# # #         fig11, ax11 = plt.subplots(figsize=(8, 4))
# # #         df['Territory Name Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax11.set_title('Territory Name Distribution')
# # #         ax11.set_ylabel('')
# # #         st.pyplot(fig11)

# # #         fig12, ax12 = plt.subplots(figsize=(8, 4))
# # #         df['Product Short'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# # #         ax12.set_title('Product Distribution')
# # #         ax12.set_ylabel('')
# # #         st.pyplot(fig12)




# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.model_selection import train_test_split, GridSearchCV
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # # from sklearn.impute import SimpleImputer
# # # from sklearn.ensemble import GradientBoostingRegressor
# # # from sklearn.metrics import mean_squared_error, r2_score
# # # import plotly.express as px  # For interactive charts

# # # # --- Helper Functions ---
# # # def create_short_form(name, max_length=4):
# # #     if isinstance(name, str):
# # #         if len(name) > max_length:
# # #             return name[:max_length] + '...'
# # #         return name
# # #     return str(name)  # Handle non-string values

# # # def create_product_short_form(name):
# # #     if isinstance(name, str):
# # #         return name.split()[0]
# # #     return str(name)  # Handle non-string values

# # # # --- Streamlit App ---
# # # st.set_page_config(layout="wide")

# # # # Add a title and a brief description
# # # st.title('Opportunity Analysis Dashboard')
# # # st.markdown("Analyze your opportunity data and gain insights with this interactive dashboard.")

# # # # File Upload
# # # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # # if uploaded_file is not None:
# # #     # Load the CSV file
# # #     try:
# # #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row
# # #     except Exception as e:
# # #         st.error(f"Error loading CSV: {e}")
# # #         st.stop()

# # #     # --- Data Preprocessing ---
# # #     df.columns = df.columns.str.strip()
# # #     df.dropna(axis=1, how='all', inplace=True)
# # #     categorical_columns = df.select_dtypes(include=['object']).columns
# # #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')
# # #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # #     for col in date_columns:
# # #         if col in df.columns:
# # #             df[col] = pd.to_datetime(df[col], errors='coerce')
# # #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# # #     imputer = SimpleImputer(strategy='mean')
# # #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
# # #     label_encoders = {}
# # #     for col in categorical_columns:
# # #         le = LabelEncoder()
# # #         df[col] = le.fit_transform(df[col])
# # #         label_encoders[col] = le
# # #     scaler = StandardScaler()
# # #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # #     # --- Feature Selection and Model Training ---
# # #     target_column = 'Amount'
# # #     features = df.drop(columns=[target_column])
# # #     target = df[target_column]
# # #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# # #     param_grid = {
# # #         'n_estimators': [50, 100, 200],
# # #         'learning_rate': [0.01, 0.1, 0.2],
# # #         'max_depth': [3, 4, 5]
# # #     }
# # #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# # #                                param_grid=param_grid,
# # #                                cv=5,
# # #                                scoring='r2',
# # #                                n_jobs=-1)
# # #     grid_search.fit(X_train, y_train)
# # #     best_params = grid_search.best_params_
# # #     best_score = grid_search.best_score_
# # #     best_model = grid_search.best_estimator_
# # #     best_model.fit(X_train, y_train)
# # #     y_pred_best = best_model.predict(X_test)
# # #     mse_best = mean_squared_error(y_test, y_pred_best)
# # #     r2_best = r2_score(y_test, y_pred_best)
# # #     feature_importances = best_model.feature_importances_
# # #     features = X_train.columns
# # #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# # #     top_5_features = importance_df.head(5)['Feature'].tolist()
# # #     for col in categorical_columns:
# # #         df[col] = label_encoders[col].inverse_transform(df[col])

# # #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# # #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# # #     # --- Interactive Design with Tabs ---
# # #     tab1, tab2, tab3 = st.tabs(["Histograms", "Line Graphs", "Pie Charts"])

# # #     with tab1:
# # #         st.header("Histograms")
# # #         col1, col2 = st.columns(2)  # Create two columns for histograms

# # #         with col1:
# # #             st.subheader('Opportunity Name Distribution')
# # #             fig1 = px.histogram(df, x='Opportunity Name Short', title='Opportunity Name Distribution')
# # #             st.plotly_chart(fig1, use_container_width=True)

# # #         with col2:
# # #             st.subheader('Territory Name Distribution')
# # #             fig2 = px.histogram(df, x='Territory Name Short', title='Territory Name Distribution')
# # #             st.plotly_chart(fig2, use_container_width=True)

# # #         st.subheader('Top 5 Features Distribution')
# # #         num_cols_hist = st.number_input("Number of features per row", min_value=1, max_value=5, value=2, key="hist_num_cols") # Add a key
# # #         columns_hist = st.columns(num_cols_hist)
# # #         num_cols_hist = len(columns_hist)

# # #         for i, feature in enumerate(top_5_features):
# # #             with columns_hist[i % num_cols_hist]:
# # #                 st.subheader(f'{feature} Distribution')
# # #                 fig = px.histogram(df, x=feature, title=f'{feature} Distribution')
# # #                 st.plotly_chart(fig, use_container_width=True)

# # #     with tab2:
# # #         st.header("Line Graphs")
# # #         st.subheader('Product Line Graph')

# # #         product_counts = df['Product Short'].value_counts().sort_index()
# # #         fig4 = px.line(x=product_counts.index, y=product_counts.values, title='Product Line Graph')
# # #         st.plotly_chart(fig4, use_container_width=True)

# # #         st.subheader('Top 5 Features Line Graphs')
# # #         num_cols_line = st.number_input("Number of features per row", min_value=1, max_value=5, value=2, key="line_num_cols") # Add a key
# # #         columns_line = st.columns(num_cols_line)
# # #         num_cols_line = len(columns_line)

# # #         for i, feature in enumerate(top_5_features):
# # #             if feature not in ['Close Date', 'Modified Date']:
# # #                 feature_counts = df[feature].value_counts().sort_index()
# # #                 with columns_line[i % num_cols_line]:
# # #                     st.subheader(f'{feature} Line Graph')
# # #                     fig5 = px.line(x=feature_counts.index, y=feature_counts.values, title=f'{feature} Line Graph')
# # #                     st.plotly_chart(fig5, use_container_width=True)

# # #     with tab3:
# # #         st.header("Pie Charts")
# # #         col1, col2 = st.columns(2)

# # #         with col1:
# # #             st.subheader('Opportunity Name Distribution')
# # #             fig9 = px.pie(df, names='Opportunity Name Short', title='Opportunity Name Distribution')
# # #             st.plotly_chart(fig9, use_container_width=True)

# # #             st.subheader('Territory Name Distribution')
# # #             fig11 = px.pie(df, names='Territory Name Short', title='Territory Name Distribution')
# # #             st.plotly_chart(fig11, use_container_width=True)

# # #         with col2:
# # #             st.subheader('Contact Person Distribution')
# # #             fig10 = px.pie(df, names='Contact Person Short', title='Contact Person Distribution')
# # #             st.plotly_chart(fig10, use_container_width=True)

# # #             st.subheader('Product Distribution')
# # #             fig12 = px.pie(df, names='Product Short', title='Product Distribution')
# # #             st.plotly_chart(fig12, use_container_width=True)





# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.impute import SimpleImputer
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.metrics import mean_squared_error, r2_score
# # import plotly.express as px  # For interactive charts

# # # --- Helper Functions ---
# # def create_short_form(name, max_length=4):
# #     if isinstance(name, str):
# #         if len(name) > max_length:
# #             return name[:max_length] + '...'
# #         return name
# #     return str(name)  # Handle non-string values

# # def create_product_short_form(name):
# #     if isinstance(name, str):
# #         return name.split()[0]
# #     return str(name)  # Handle non-string values

# # # --- Streamlit App ---
# # st.set_page_config(layout="wide")

# # # Add a title and a brief description
# # st.title('Opportunity Analysis Dashboard')
# # st.markdown("Analyze your opportunity data and gain insights with this interactive dashboard.")

# # # File Upload
# # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     # Load the CSV file
# #     try:
# #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row
# #     except Exception as e:
# #         st.error(f"Error loading CSV: {e}")
# #         st.stop()

# #     # --- Data Preprocessing ---
# #     df.columns = df.columns.str.strip()
# #     df.dropna(axis=1, how='all', inplace=True)
# #     categorical_columns = df.select_dtypes(include=['object']).columns
# #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')
# #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# #     for col in date_columns:
# #         if col in df.columns:
# #             df[col] = pd.to_datetime(df[col], errors='coerce')
# #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# #     imputer = SimpleImputer(strategy='mean')
# #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
# #     label_encoders = {}
# #     for col in categorical_columns:
# #         le = LabelEncoder()
# #         df[col] = le.fit_transform(df[col])
# #         label_encoders[col] = le
# #     scaler = StandardScaler()
# #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# #     # --- Feature Selection and Model Training ---
# #     target_column = 'Amount'
# #     features = df.drop(columns=[target_column])
# #     target = df[target_column]
# #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# #     param_grid = {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'max_depth': [3, 4, 5]
# #     }
# #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# #                                param_grid=param_grid,
# #                                cv=5,
# #                                scoring='r2',
# #                                n_jobs=-1)
# #     grid_search.fit(X_train, y_train)
# #     best_params = grid_search.best_params_
# #     best_score = grid_search.best_score_
# #     best_model = grid_search.best_estimator_
# #     best_model.fit(X_train, y_train)
# #     y_pred_best = best_model.predict(X_test)
# #     mse_best = mean_squared_error(y_test, y_pred_best)
# #     r2_best = r2_score(y_test, y_pred_best)
# #     feature_importances = best_model.feature_importances_
# #     features = X_train.columns
# #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# #     top_5_features = importance_df.head(3)['Feature'].tolist() # only show top 3 to prevent memory issues
# #     for col in categorical_columns:
# #         df[col] = label_encoders[col].inverse_transform(df[col])

# #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# #     # --- Interactive Design with Tabs ---
# #     tab1, tab2, tab3 = st.tabs(["Histograms", "Line Graphs", "Pie Charts"])

# #     with tab1:
# #         st.header("Histograms")
# #         col1, col2 = st.columns(2)  # Create two columns for histograms

# #         with col1:
# #             st.subheader('Opportunity Name Distribution')
# #             fig1 = px.histogram(df, x='Opportunity Name Short', title='Opportunity Name Distribution')
# #             st.plotly_chart(fig1, use_container_width=True)

# #         with col2:
# #             st.subheader('Territory Name Distribution')
# #             fig2 = px.histogram(df, x='Territory Name Short', title='Territory Name Distribution')
# #             st.plotly_chart(fig2, use_container_width=True)

# #         st.subheader('Top 5 Features Distribution')
# #         num_cols_hist = st.number_input("Number of features per row", min_value=1, max_value=5, value=2, key="hist_num_cols") # Add a key
# #         columns_hist = st.columns(num_cols_hist)
# #         num_cols_hist = len(columns_hist)

# #         for i, feature in enumerate(top_5_features):
# #             with columns_hist[i % num_cols_hist]:
# #                 st.subheader(f'{feature} Distribution')
# #                 fig = px.histogram(df, x=feature, title=f'{feature} Distribution')
# #                 st.plotly_chart(fig, use_container_width=True)

# #     with tab2:
# #         st.header("Line Graphs")
# #         col1, col2, col3 = st.columns(3) #Dividing three columns

# #         with col1:
# #             st.subheader('Product Line Graph')
# #             product_counts = df['Product Short'].value_counts().sort_index()
# #             fig4 = px.line(x=product_counts.index, y=product_counts.values, title='Product Line Graph')
# #             st.plotly_chart(fig4, use_container_width=True)

# #         #Iterate through top 3 feature in line graph
# #         for i, feature in enumerate(top_5_features):
# #             if feature not in ['Close Date', 'Modified Date']:
# #                 feature_counts = df[feature].value_counts().sort_index()
# #                 if i == 0:
# #                     with col2:
# #                         st.subheader(f'{feature} Line Graph')
# #                         fig5 = px.line(x=feature_counts.index, y=feature_counts.values, title=f'{feature} Line Graph')
# #                         st.plotly_chart(fig5, use_container_width=True)
# #                 elif i == 1:
# #                     with col3:
# #                         st.subheader(f'{feature} Line Graph')
# #                         fig5 = px.line(x=feature_counts.index, y=feature_counts.values, title=f'{feature} Line Graph')
# #                         st.plotly_chart(fig5, use_container_width=True)
# #                 else:
# #                     break # only show 3 feature to prevent too many graph

# #     with tab3:
# #         st.header("Pie Charts")
# #         col1, col2 = st.columns(2)

# #         with col1:
# #             st.subheader('Opportunity Name Distribution')
# #             fig9 = px.pie(df, names='Opportunity Name Short', title='Opportunity Name Distribution')
# #             st.plotly_chart(fig9, use_container_width=True)

# #             st.subheader('Territory Name Distribution')
# #             fig11 = px.pie(df, names='Territory Name Short', title='Territory Name Distribution')
# #             st.plotly_chart(fig11, use_container_width=True)

# #         with col2:
# #             st.subheader('Contact Person Distribution')
# #             fig10 = px.pie(df, names='Contact Person Short', title='Contact Person Distribution')
# #             st.plotly_chart(fig10, use_container_width=True)

# #             st.subheader('Product Distribution')
# #             fig12 = px.pie(df, names='Product Short', title='Product Distribution')
# #             st.plotly_chart(fig12, use_container_width=True)



# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.impute import SimpleImputer
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.metrics import mean_squared_error, r2_score
# # import plotly.express as px  # For interactive charts

# # # --- Helper Functions ---
# # def create_short_form(name, max_length=4):
# #     if isinstance(name, str):
# #         if len(name) > max_length:
# #             return name[:max_length] + '...'
# #         return name
# #     return str(name)  # Handle non-string values

# # def create_product_short_form(name):
# #     if isinstance(name, str):
# #         return name.split()[0]
# #     return str(name)  # Handle non-string values

# # # --- Streamlit App ---
# # st.set_page_config(layout="wide")

# # # Add a title and a brief description
# # st.title('Opportunity Analysis Dashboard')
# # st.markdown("Analyze your opportunity data and gain insights with this interactive dashboard.")

# # # File Upload
# # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     # Load the CSV file
# #     try:
# #         df = pd.read_csv(uploaded_file, skiprows=1)  # Skip the first row
# #     except Exception as e:
# #         st.error(f"Error loading CSV: {e}")
# #         st.stop()

# #     # --- Data Preprocessing ---
# #     df.columns = df.columns.str.strip()
# #     df.dropna(axis=1, how='all', inplace=True)
# #     categorical_columns = df.select_dtypes(include=['object']).columns
# #     df[categorical_columns] = df[categorical_columns].fillna('Unknown')
# #     date_columns = ['Created Date', 'Modified Date', 'Close Date']
# #     for col in date_columns:
# #         if col in df.columns:
# #             df[col] = pd.to_datetime(df[col], errors='coerce')
# #     numeric_columns = df.select_dtypes(include=[np.number]).columns
# #     imputer = SimpleImputer(strategy='mean')
# #     df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
# #     label_encoders = {}
# #     for col in categorical_columns:
# #         le = LabelEncoder()
# #         df[col] = le.fit_transform(df[col])
# #         label_encoders[col] = le
# #     scaler = StandardScaler()
# #     df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# #     # --- Feature Selection and Model Training ---
# #     target_column = 'Amount'
# #     features = df.drop(columns=[target_column])
# #     target = df[target_column]
# #     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# #     param_grid = {
# #         'n_estimators': [50, 100, 200],
# #         'learning_rate': [0.01, 0.1, 0.2],
# #         'max_depth': [3, 4, 5]
# #     }
# #     grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# #                                param_grid=param_grid,
# #                                cv=5,
# #                                scoring='r2',
# #                                n_jobs=-1)
# #     grid_search.fit(X_train, y_train)
# #     best_params = grid_search.best_params_
# #     best_score = grid_search.best_score_
# #     best_model = grid_search.best_estimator_
# #     best_model.fit(X_train, y_train)
# #     y_pred_best = best_model.predict(X_test)
# #     mse_best = mean_squared_error(y_test, y_pred_best)
# #     r2_best = r2_score(y_test, y_pred_best)
# #     feature_importances = best_model.feature_importances_
# #     features = X_train.columns
# #     importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# #     importance_df = importance_df.sort_values(by='Importance', ascending=False)
# #     top_5_features = importance_df.head(5)['Feature'].tolist()
# #     for col in categorical_columns:
# #         df[col] = label_encoders[col].inverse_transform(df[col])

# #     df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# #     df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# #     df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# #     df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# #     df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0] if isinstance(x, str) else str(x))
# #     df['Product Short'] = df['Product'].apply(create_product_short_form)

# #     # --- Interactive Design with Tabs ---
# #     tab1, tab2, tab3 = st.tabs(["Histograms", "Line Graphs", "Pie Charts"])

# #     with tab1:
# #         st.header("Histograms")
# #         col1, col2 = st.columns(2)  # Create two columns for histograms

# #         with col1:
# #             st.subheader('Opportunity Name Distribution')
# #             fig1 = px.histogram(df, x='Opportunity Name Short', title='Opportunity Name Distribution')
# #             st.plotly_chart(fig1, use_container_width=True)

# #         with col2:
# #             st.subheader('Territory Name Distribution')
# #             fig2 = px.histogram(df, x='Territory Name Short', title='Territory Name Distribution')
# #             st.plotly_chart(fig2, use_container_width=True)

# #         st.subheader('Top 5 Features Distribution')
# #         num_cols_hist = st.number_input("Number of features per row", min_value=1, max_value=5, value=2, key="hist_num_cols") # Add a key
# #         columns_hist = st.columns(num_cols_hist)
# #         num_cols_hist = len(columns_hist)

# #         for i, feature in enumerate(top_5_features):
# #             with columns_hist[i % num_cols_hist]:
# #                 st.subheader(f'{feature} Distribution')
# #                 fig = px.histogram(df, x=feature, title=f'{feature} Distribution')
# #                 st.plotly_chart(fig, use_container_width=True)

# #     with tab2:
# #         st.header("Line Graphs")

# #         st.subheader('Product Line Graph')
# #         product_counts = df['Product Short'].value_counts().sort_index()
# #         fig4 = px.line(x=product_counts.index, y=product_counts.values, title='Product Line Graph')
# #         st.plotly_chart(fig4, use_container_width=True)

# #         st.subheader('Top 5 Features Line Graphs')
# #         num_cols_line = st.number_input("Number of features per row", min_value=1, max_value=5, value=2, key="line_num_cols")
# #         columns_line = st.columns(num_cols_line)
# #         num_cols_line = len(columns_line)

# #         for i, feature in enumerate(top_5_features):
# #             if feature not in ['Close Date', 'Modified Date']:
# #                 feature_counts = df[feature].value_counts().sort_index()
# #                 with columns_line[i % num_cols_line]:
# #                     st.subheader(f'{feature} Line Graph')
# #                     fig5 = px.line(x=feature_counts.index, y=feature_counts.values, title=f'{feature} Line Graph')
# #                     st.plotly_chart(fig5, use_container_width=True)

# #     with tab3:
# #         st.header("Pie Charts")
# #         col1, col2 = st.columns(2)

# #         with col1:
# #             st.subheader('Opportunity Name Distribution')
# #             fig9 = px.pie(df, names='Opportunity Name Short', title='Opportunity Name Distribution')
# #             st.plotly_chart(fig9, use_container_width=True)

# #             st.subheader('Territory Name Distribution')
# #             fig11 = px.pie(df, names='Territory Name Short', title='Territory Name Distribution')
# #             st.plotly_chart(fig11, use_container_width=True)

# #         with col2:
# #             st.subheader('Contact Person Distribution')
# #             fig10 = px.pie(df, names='Contact Person Short', title='Contact Person Distribution')
# #             st.plotly_chart(fig10, use_container_width=True)

# #             st.subheader('Product Distribution')
# #             fig12 = px.pie(df, names='Product Short', title='Product Distribution')
# #             st.plotly_chart(fig12, use_container_width=True)






# # import os
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.impute import SimpleImputer
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.metrics import mean_squared_error, r2_score
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import plotly.express as px

# # # Create a directory to save images
# # image_dir = 'image1'
# # if not os.path.exists(image_dir):
# #     os.makedirs(image_dir)

# # # Load the CSV file
# # file_path = 'D:/skyward/Opportunity Details By Owner (27).csv'
# # df = pd.read_csv(file_path, skiprows=1)  # Skip the first row which seems to be empty

# # # Display the first few rows
# # print(df.head())

# # # Strip leading and trailing spaces from column names
# # df.columns = df.columns.str.strip()

# # # Check the actual column names
# # print("Column names:", df.columns.tolist())

# # # Data Preprocessing
# # # Drop columns with all missing values
# # df.dropna(axis=1, how='all', inplace=True)

# # # Fill missing values for categorical columns with 'Unknown'
# # categorical_columns = df.select_dtypes(include=['object']).columns
# # df[categorical_columns] = df[categorical_columns].fillna('Unknown')

# # # Convert date columns to datetime
# # date_columns = ['Created Date', 'Modified Date', 'Close Date']
# # for col in date_columns:
# #     if col in df.columns:
# #         df[col] = pd.to_datetime(df[col], errors='coerce')

# # # Fill missing values for numerical columns with the mean
# # numeric_columns = df.select_dtypes(include=[np.number]).columns
# # imputer = SimpleImputer(strategy='mean')
# # df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# # # Encode categorical columns
# # label_encoders = {}
# # for col in categorical_columns:
# #     le = LabelEncoder()
# #     df[col] = le.fit_transform(df[col])
# #     label_encoders[col] = le

# # # Scale numerical data
# # scaler = StandardScaler()
# # df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # # Feature Selection and Target Variable
# # # For this example, let's predict 'Amount'
# # target_column = 'Amount'
# # features = df.drop(columns=[target_column])
# # target = df[target_column]

# # # Train-test split
# # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # Hyperparameter tuning using GridSearchCV
# # param_grid = {
# #     'n_estimators': [50, 100, 200],
# #     'learning_rate': [0.01, 0.1, 0.2],
# #     'max_depth': [3, 4, 5]
# # }

# # grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
# #                            param_grid=param_grid,
# #                            cv=5,
# #                            scoring='r2',
# #                            n_jobs=-1)

# # grid_search.fit(X_train, y_train)

# # # Best parameters and best score
# # best_params = grid_search.best_params_
# # best_score = grid_search.best_score_
# # print(f"Best Parameters: {best_params}")
# # print(f"Best Cross-Validation R^2 Score: {best_score}")

# # # Train model with best parameters
# # best_model = grid_search.best_estimator_
# # best_model.fit(X_train, y_train)

# # # Make predictions with the best model
# # y_pred_best = best_model.predict(X_test)

# # # Evaluate the best model
# # mse_best = mean_squared_error(y_test, y_pred_best)
# # r2_best = r2_score(y_test, y_pred_best)
# # print(f"Mean Squared Error (Best Model): {mse_best}")
# # print(f"R^2 Score (Best Model): {r2_best}")

# # # Provide insights for new customers
# # feature_importances = best_model.feature_importances_
# # features = X_train.columns
# # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
# # importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # print("\nTop 5 Features that influence the Amount:")
# # top_5_features = importance_df.head(5)['Feature'].tolist()
# # print(importance_df.head(10))

# # # Decode categorical features back to their original labels
# # for col in categorical_columns:
# #     df[col] = label_encoders[col].inverse_transform(df[col])

# # # Extract only the name from the contact person field
# # df['Contact Person'] = df['Contact Person'].apply(lambda x: x.split(' - ')[0])

# # # Create short forms for long names
# # def create_short_form(name, max_length=4):
# #     if len(name) > max_length:
# #         return name[:max_length] + '...'
# #     return name

# # # Create short forms for product names by taking only the first word
# # df['Product'] = df['Product'].apply(lambda x: x.split(' - ')[0])
# # def create_product_short_form(name):
# #     return name.split()[0]

# # # Apply short forms to specific columns
# # df['Opportunity Name Short'] = df['Opportunity Name'].apply(create_short_form)
# # df['Territory Name Short'] = df['Territory Name'].apply(create_short_form)
# # df['Contact Person Short'] = df['Contact Person'].apply(create_short_form)
# # df['Product Short'] = df['Product'].apply(create_product_short_form)
# # df['Owner Short'] = df['Owner'].apply(create_short_form)
# # df['Source Short'] = df['Source'].apply(create_short_form)
# # df['Customer Name Short'] = df['Customer Name'].apply(create_short_form)
# # df['Industry Short'] = df['Industry'].apply(create_short_form)

# # # Plot histograms for Opportunity Name, Territory Name, Owner, Source, Customer Name, Industry, and Top 5 Features using short forms
# # columns_to_plot = ['Opportunity Name Short', 'Territory Name Short', 'Owner Short', 'Source Short', 'Customer Name Short', 'Industry Short']

# # for column in columns_to_plot:
# #     plt.figure(figsize=(10, 6))
# #     sns.histplot(df[column], kde=True)
# #     plt.title(f'{column} Distribution')
# #     plt.xlabel(column)
# #     plt.ylabel('Frequency')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(image_dir, f'{column.lower()}_distribution_histogram.png'))
# #     plt.close()

# # for feature in top_5_features:
# #     plt.figure(figsize=(10, 6))
# #     sns.histplot(df[feature], kde=True)
# #     plt.title(f'{feature} Distribution')
# #     plt.xlabel(feature)
# #     plt.ylabel('Frequency')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(image_dir, f'{feature}_distribution_histogram.png'))
# #     plt.close()

# # # Plot line graphs for Product, Owner, Source, Customer Name, Industry, and Top 5 Features using short forms
# # columns_to_plot = ['Product Short', 'Owner Short', 'Source Short', 'Customer Name Short', 'Industry Short']

# # for column in columns_to_plot:
# #     counts = df[column].value_counts().sort_index()
# #     plt.figure(figsize=(10, 6))
# #     sns.lineplot(x=counts.index, y=counts.values)
# #     plt.title(f'{column} Line Graph')
# #     plt.xlabel(column)
# #     plt.ylabel('Frequency')
# #     plt.xticks(rotation=45, ha='right')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(image_dir, f'{column.lower()}_line_graph.png'))
# #     plt.close()

# # for feature in top_5_features:
# #     if feature not in ['Close Date', 'Modified Date']:
# #         feature_counts = df[feature].value_counts().sort_index()
# #         plt.figure(figsize=(10, 6))
# #         sns.lineplot(x=feature_counts.index, y=feature_counts.values)
# #         plt.title(f'{feature} Line Graph')
# #         plt.xlabel(feature)
# #         plt.ylabel('Frequency')
# #         plt.xticks(rotation=45, ha='right')
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(image_dir, f'{feature}_line_graph.png'))
# #         plt.close()

# # # Plot pie charts for Opportunity Name, Territory Name, Owner, Source, Customer Name, Industry, Product, and Top 5 Features using short forms
# # columns_to_plot = ['Opportunity Name Short', 'Territory Name Short', 'Owner Short', 'Source Short', 'Customer Name Short', 'Industry Short', 'Product Short']

# # for column in columns_to_plot:
# #     plt.figure(figsize=(10, 6))
# #     df[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
# #     plt.title(f'{column} Distribution')
# #     plt.ylabel('')
# #     plt.tight_layout()
# #     plt.savefig(os.path.join(image_dir, f'{column.lower()}_distribution_pie.png'))
# #     plt.close()

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.express as px
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.impute import SimpleImputer

# # # Streamlit UI
# # st.title("📊 Business Data Visualization Dashboard")

# # # File uploader
# # uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     # Read the CSV file
# #     df = pd.read_csv(uploaded_file, skiprows=1)  # Skip first row if needed
# #     df.columns = df.columns.str.strip()  # Remove spaces from column names

# #     # Debugging: Display column names
    

# #     # Ensure "Amount" column exists
# #     if "Amount" in df.columns:
# #         df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")  # Convert to numeric

# #         # Select categorical column
# #         category = st.selectbox(
# #             "📌 Select Category for Visualization",
# #             ["Owner", "Customer Name", "Territory Name", "Source", "Industry", "Product"]
# #         )

# #         # Select chart type
# #         chart_type = st.selectbox(
# #             "📊 Select Chart Type",
# #             ["Histogram", "Pie Chart", "Line Chart"]
# #         )

# #         # Displaying Graphs
# #         if category in df.columns:
# #             grouped_df = df.groupby(category, as_index=False)["Amount"].sum()

# #             if chart_type == "Histogram":
# #                 fig = px.bar(grouped_df, x=category, y="Amount", title=f"{category} vs Amount")
# #             elif chart_type == "Pie Chart":
# #                 # **Limit to top 10 categories to avoid clutter**
# #                 top_categories = grouped_df.nlargest(10, "Amount")  
# #                 fig = px.pie(
# #                     top_categories, 
# #                     names=category, 
# #                     values="Amount", 
# #                     title=f"{category} Distribution", 
# #                     hole=0.3,  # Creates a donut-style chart
# #                     color_discrete_sequence=px.colors.sequential.RdBu,
# #                 )
# #                 fig.update_traces(textinfo='percent+label')  # Show percentage & label
# #             elif chart_type == "Line Chart":
# #                 fig = px.line(grouped_df, x=category, y="Amount", title=f"{category} vs Amount", markers=True)

# #             st.plotly_chart(fig)
# #         else:
# #             st.error(f"⚠️ Selected category '{category}' is missing in the dataset.")

# #         # ---- TOP 5 FEATURE ANALYSIS ----
        
# #     else:
# #         st.error("⚠️ The 'Amount' column is missing. Please check the CSV file.")

# # else:
# #     st.info("📂 Please upload a CSV file to proceed.")





# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# # Streamlit UI
# st.title("📊 Business Data Visualization Dashboard")

# # File uploader
# uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

# if uploaded_file is not None:
#     # Read the CSV file
#     df = pd.read_csv(uploaded_file, skiprows=1)  # Skip first row if needed
#     df.columns = df.columns.str.strip()  # Remove spaces from column names

#     # Ensure "Amount" column exists
#     if "Amount" in df.columns:
#         df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")  # Convert to numeric

#         # Select categorical column
#         category = st.selectbox(
#             "📌 Select Category for Visualization",
#             ["Owner", "Customer Name", "Territory Name", "Source", "Industry", "Product"]
#         )

#         # Select chart type
#         chart_type = st.selectbox(
#             "📊 Select Chart Type",
#             ["Histogram", "Pie Chart", "Line Chart"]
#         )

#         # Displaying Graphs
#         if category in df.columns:
#             grouped_df = df.groupby(category, as_index=False)["Amount"].sum()

#             if chart_type == "Histogram":
#                 fig = px.bar(grouped_df, x=category, y="Amount", title=f"{category} vs Amount")

#             elif chart_type == "Pie Chart":
#                 # **Limit to top 10 categories to avoid clutter**
#                 top_categories = grouped_df.nlargest(10, "Amount")  

#                 fig = px.pie(
#                     top_categories, 
#                     names=category, 
#                     values="Amount", 
#                     title=f"{category} Distribution", 
#                     hole=0.3,  # Donut-style chart
#                     color_discrete_sequence=px.colors.sequential.RdBu,
#                 )
#                 fig.update_traces(
#                     textinfo='percent',  # Show only percentages
#                     pull=[0.1] * len(top_categories),  # Pull out slices slightly
#                     textposition='outside',  # Move labels outside
#                     insidetextorientation='horizontal'  # Avoid text rotation
#                 )
#                 fig.update_layout(
#                     showlegend=True,  # Keep legend
#                     margin=dict(t=40, b=20, l=20, r=20)  # Adjust spacing
#                 )

#             elif chart_type == "Line Chart":
#                 fig = px.line(grouped_df, x=category, y="Amount", title=f"{category} vs Amount", markers=True)

#             st.plotly_chart(fig)

#         else:
#             st.error(f"⚠️ Selected category '{category}' is missing in the dataset.")

#     else:
#         st.error("⚠️ The 'Amount' column is missing. Please check the CSV file.")

# else:
#     st.info("📂 Please upload a CSV file to proceed.")




import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Streamlit UI
st.title("📊 Business Data Visualization Dashboard")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file, skiprows=1)  # Skip first row if needed
    df.columns = df.columns.str.strip()  # Remove spaces from column names

    # Ensure "Amount" column exists
    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")  # Convert to numeric

        # Select categorical column
        category = st.selectbox(
            "📌 Select Category for Visualization",
            ["Owner", "Customer Name", "Territory Name", "Source", "Industry", "Product"]
        )

        # Select chart type
        chart_type = st.selectbox(
            "📊 Select Chart Type",
            ["Histogram", "Pie Chart", "Line Chart"]
        )

        # Displaying Graphs
        if category in df.columns:
            grouped_df = df.groupby(category, as_index=False)["Amount"].sum()

            if chart_type == "Histogram":
                fig = px.bar(grouped_df, x=category, y="Amount", title=f"{category} vs Amount")

            elif chart_type == "Pie Chart":
                # **Limit to top 10 categories to avoid clutter**
                top_categories = grouped_df.nlargest(10, "Amount")  

                # Shorten long names for display, keep full name in hover tooltip
                top_categories["Short_Label"] = top_categories[category].apply(lambda x: x[:15] + "..." if len(x) > 15 else x)

                fig = px.pie(
                    top_categories, 
                    names="Short_Label",  # Display shorter labels
                    values="Amount", 
                    title=f"{category} Distribution", 
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_traces(
                    textinfo='percent',  # Show both percent and label
                    textposition='outside',   # Place labels outside
                    hovertemplate=f"{category}: %{{label}}<br>Amount: %{{value}}"  # Keep full names in hover tooltip
                )
                fig.update_layout(
                    showlegend=True, 
                    height=600,  # Increase figure size
                    margin=dict(t=50, b=30, l=50, r=50)  # Adjust margins
                )

            elif chart_type == "Line Chart":
                fig = px.line(grouped_df, x=category, y="Amount", title=f"{category} vs Amount", markers=True)

            st.plotly_chart(fig)

        else:
            st.error(f"⚠️ Selected category '{category}' is missing in the dataset.")

    else:
        st.error("⚠️ The 'Amount' column is missing. Please check the CSV file.")

else:
    st.info("📂 Please upload a CSV file to proceed.")
