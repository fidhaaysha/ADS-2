# Import the required libraries
import pandas as pd
from scipy import stats

# Function to process the dataset
def process_dataset(filename):
    # Read the dataset
    dataset = pd.read_csv(filename)
    
    # Remove leading/trailing whitespaces in column names
    dataset.columns = dataset.columns.str.strip()
    
    # Transpose the dataset
    country_transposed_data = dataset.set_index('Country Name')
    year_transposed_data = dataset.transpose()
    
    # Display basic statistics and the first 25 rows of the dataset
    print(dataset.describe())
    print(dataset.head(25))
    
    return dataset, country_transposed_data, year_transposed_data

# Example usage
processed_data, country_transposed_data, year_transposed_data = process_dataset("Renewable Energy Adoption and Economic Growth.csv")

# List of countries of interest
country_of_interest = ['China', 'India', 'Australia','Brazil','Australia','Bangladesh']

# Filter the dataset for the countries of interest
filtered_data = processed_data[processed_data['Country Name'].isin(country_of_interest)]

# List of relevant columns for analysis
selected_columns = ['Country Name', 'GDP growth (annual %)', 'Energy use (kg of oil equivalent per capita)',
                     'Renewable energy consumption (% of total final energy consumption)',
                     'Methane emissions in the energy sector (thousand metric tons of CO2 equivalent)']

# Filter the dataset for selected columns and countries
selected_data = processed_data[(processed_data['Country Name'].isin(country_of_interest)) & 
                                (processed_data['Indicator Name'].isin(selected_columns))]

# Display summary statistics
summary_statistics = selected_data.describe()
print(summary_statistics)

# Function to compute skewness and kurtosis
def statistics(dataframe):
    """This function takes the dataframe as an argument and finds the skewness
    and kurtosis of the distribution."""
    print("Skewness = ")
    print(stats.skew(dataframe))
    print("Kurtosis = ")
    print(stats.kurtosis(dataframe))

# Function to filter and analyze data
def filter_and_analyze(data, countries, indicators):
    """This function filters the data for specific countries and indicators,
    then performs statistical analysis."""
    filtered_data = data[(data['Country Name'].isin(countries)) & 
                         (data['Indicator Name'].isin(indicators))]
    
    # Replace non-numeric values with NaN
    filtered_data = filtered_data.replace('..', pd.NA)
    
    # Display summary statistics
    print("\nSummary Statistics:")
    summary_statistics = filtered_data.groupby(['Country Name', 'Indicator Name']).describe().transpose()
    print(summary_statistics)
    
    # Normalize the data for selected indicators
    normalized_data = filtered_data.pivot_table(index='Country Name', columns='Indicator Name', values='2022', aggfunc='first')
    
    print("\nNormalized Data (Before NaN check):")
    print(normalized_data)
    
    normalized_data = normalized_data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
    normalized_data = (normalized_data - normalized_data.mean()) / normalized_data.std()

    # Display normalized data statistics
    print("\nNormalized Data Statistics:")
    normalized_statistics = normalized_data.describe()
    print(normalized_statistics)

    # Call the statistics function for each country
    for country in countries:
        country_data = normalized_data.loc[country]
        print(f"\nSkewness and Kurtosis for {country}:")
        statistics(country_data)

# Example usage
countries_of_interest = ['United States', 'China', 'India', 'Brazil']
selected_indicators = ['Population growth (annual %)',
                        'Alternative and nuclear energy (% of total energy use)',
                        'Nitrous oxide emissions in energy sector (thousand metric tons of CO2 equivalent)',
                        'CO2 intensity (kg per kg of oil equivalent energy use)']

# Assuming `processed_data` is the dataframe you obtained from processing the dataset
filter_and_analyze(processed_data, countries_of_interest, selected_indicators)














#Renewable Energy Adoption (% of total final energy consumption - Bar graph

# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the time period of interest (1998 to 2022)
renewable_data = processed_data[
    (processed_data['Country Name'].isin(country_of_interest)) &
    (processed_data['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')
]

# Extract relevant columns for plotting
renewable_data = renewable_data[['Country Name'] + list(map(str, range(1998, 2023, 5)))]

# Set 'Country Name' as the index for plotting
renewable_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
renewable_data = renewable_data.apply(pd.to_numeric, errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
renewable_data.transpose().plot(kind='bar', width=0.8)
plt.title('Renewable Energy Adoption of Countries in 5-Year Intervals')
plt.xlabel('Year')
plt.ylabel('Renewable Energy Adoption (% of total final energy consumption)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()








#GDP Growth of Countries (1998-2022) - Bar graph

# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the time period of interest (1998 to 2022)
gdp_data = processed_data[(processed_data['Country Name'].isin(country_of_interest)) & 
                           (processed_data['Indicator Name'] == 'GDP growth (annual %)')]

# Extract relevant columns for plotting
gdp_data = gdp_data[['Country Name'] + list(map(str, range(1998, 2023, 5)))]

# Set 'Country Name' as the index for plotting
gdp_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
gdp_data = gdp_data.apply(pd.to_numeric, errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
gdp_data.transpose().plot(kind='bar', width=0.8)
plt.title('GDP Growth of Countries in 5-Year Intervals')
plt.xlabel('Year')
plt.ylabel('GDP Growth (annual %)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()








#Correlation Heatmap:Energy and Population of Bangladesh

# Import the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for Bangladesh
bangladesh_data = processed_data[processed_data['Country Name'] == 'Bangladesh']

# Extract relevant columns for correlation heatmap
correlation_data = bangladesh_data[bangladesh_data['Indicator Name'].isin([
    'Renewable energy consumption (% of total final energy consumption)',
    'Renewable energy consumption (thousand metric tons of oil equivalent)',
    'Population growth (annual %)',
    'Urban population growth (annual %)',
    'Energy use (kg of oil equivalent per capita)',
    'Rural population growth (annual %)'
])]

# Pivot the data to have years as columns
correlation_data_pivot = correlation_data.pivot(index='Indicator Name', columns='Country Name', values=correlation_data.columns[4:])

# Convert data to numeric (in case it's not)
correlation_data_pivot = correlation_data_pivot.apply(pd.to_numeric, errors='coerce')

# Compute the correlation matrix
correlation_matrix = correlation_data_pivot.transpose().corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap showing Energy VS Population of Bangladesh')
plt.show()



#Pie chart for Emissions Distribution in the United Kingdom

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the United Kingdom
uk_data = processed_data[processed_data['Country Name'] == 'United Kingdom']

# Extract relevant columns for the pie chart
pie_chart_data = uk_data[uk_data['Indicator Name'].isin([
    'Methane emissions in energy sector (thousand metric tons of CO2 equivalent)',
    'Nitrous oxide emissions in energy sector (thousand metric tons of CO2 equivalent)'
])]

# Set the 'Indicator Name' as the index for the pie chart data
pie_chart_data.set_index('Indicator Name', inplace=True)

# Convert the values to numeric
pie_chart_data = pie_chart_data.apply(pd.to_numeric, errors='coerce')

# Extract the values for the pie chart
values = pie_chart_data.loc[:, '1998':'2022'].sum(axis=1)

# Set custom colors
custom_colors = ['#66c2a5', '#fc8d62']  # Different color codes

# Plot the pie chart with custom colors
plt.figure(figsize=(8, 8))
plt.pie(values, labels=values.index, autopct='%1.1f%%', startangle=140, colors=custom_colors)
plt.title('Emissions Distribution in the United Kingdom (1998-2022)')
plt.show()









#Table: Urban Population Growth (%)

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for the time period of interest (1998 to 2022)
urban_population_data = processed_data[
    (processed_data['Indicator Name'] == 'Urban population growth (annual %)') &
    (processed_data['Country Name'] != 'World')  # Exclude 'World' row if present
]

# Extract relevant columns for the table
urban_population_table = urban_population_data[['Country Name'] + list(map(str, range(1998, 2023, 10)))]

# Set 'Country Name' as the index for the table
urban_population_table.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
urban_population_table = urban_population_table.apply(pd.to_numeric, errors='coerce')

# Display the table
print("Urban Population Growth (%) - 10-Year Intervals (1998-2022)\n")
print(urban_population_table)











#Heat Map for Energy Emissions in Brazil

# Import the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Filter the dataset for Brazil
brazil_data = processed_data[processed_data['Country Name'] == 'Brazil']

# Extract relevant columns for the heatmap
heatmap_data_brazil = brazil_data[brazil_data['Indicator Name'].isin([
    'Methane emissions in energy sector (thousand metric tons of CO2 equivalent)',
    'Nitrous oxide emissions in energy sector (% of total)',
    'Energy related methane emissions (% of total)',
    'CO2 intensity (kg per kg of oil equivalent energy use)',
    'Nitrous oxide emissions in energy sector (thousand metric tons of CO2 equivalent)'
])]

# Set the 'Indicator Name' as the index for the heatmap
heatmap_data_brazil.set_index('Indicator Name', inplace=True)

# Extract the year columns for the heatmap
heatmap_data_brazil = heatmap_data_brazil.set_index('Country Name').filter(regex=r'^\d{4}$')

# Convert data to numeric (in case it's not)
heatmap_data_brazil = heatmap_data_brazil.apply(pd.to_numeric, errors='coerce')

# Create a correlation matrix
correlation_matrix_brazil = heatmap_data_brazil.transpose().corr()

# Set a custom color palette with black, grey, and green
custom_palette = sns.color_palette(['black', 'grey', 'green'], as_cmap=True)

# Plot the heatmap with the custom color palette
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_brazil, annot=True, cmap=custom_palette, fmt=".2f", linewidths=.5)

# Customize x-axis and y-axis labels with explicit indicator names
indicator_names = [
    'Methane emissions in energy sector (thousand metric tons of CO2 equivalent)',
    'Nitrous oxide emissions in energy sector (% of total)',
    'Energy related methane emissions (% of total)',
    'CO2 intensity (kg per kg of oil equivalent energy use)',
    'Nitrous oxide emissions in energy sector (thousand metric tons of CO2 equivalent)',
]
plt.title('Correlation Heatmap showing Energy Emissions in Brazil')
plt.xlabel('Indicators')
plt.ylabel('Indicators')
plt.xticks(ticks=range(len(indicator_names)), labels=indicator_names, rotation=45)
plt.yticks(ticks=range(len(indicator_names)), labels=indicator_names, rotation=0)

# Show the plot
plt.show()






#Renewable Energy Consumption - Line graph

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Define countries of interest
countries_of_interest = ['United States', 'China', 'India', 'Australia','Brazil','Australia','South Africa','Bangladesh']

# Filter the dataset for the time period of interest (1998 to 2022) and countries of interest
renewable_data = processed_data[
    (processed_data['Country Name'].isin(countries_of_interest)) &
    (processed_data['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')
]

# Extract relevant columns for plotting
renewable_data = renewable_data[['Country Name'] + list(map(str, range(1998, 2023, 5)))]

# Set 'Country Name' as the index for plotting
renewable_data.set_index('Country Name', inplace=True)

# Convert data to numeric (in case it's not)
renewable_data = renewable_data.apply(pd.to_numeric, errors='coerce')

# Plot the data using seaborn lineplot
plt.figure(figsize=(12, 6))
sns.lineplot(data=renewable_data.transpose(), markers=True, dashes=False)
plt.title('Renewable Energy Consumption of Countries (1998-2022) in 5-Year Intervals')
plt.xlabel('Year')
plt.ylabel('Renewable Energy Consumption (% of total final energy)')
plt.xticks(rotation=45)
plt.legend(title='Country')
plt.tight_layout()
plt.show()











#Correlation Heatmap of Alternate Energy in India

# Import the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming processed_data is the processed dataset from your previous script
# If not, make sure to include the necessary code to read and process the data

# Display the unique column names in the dataset
print(processed_data['Indicator Name'].unique())

# Filter the dataset for India and renewable energy adoption
india_renewable_data = processed_data[
    (processed_data['Country Name'] == 'India') &
    (processed_data['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)')
]

# Extract relevant columns for correlation analysis
correlation_data = processed_data[
    (processed_data['Country Name'].isin(['India'])) &
    (processed_data['Indicator Name'].isin([
        'GDP growth (annual %)',
        'Energy use (kg of oil equivalent per capita)',
        'Renewable energy consumption (% of total final energy consumption)',
        'Methane emissions in the energy sector (thousand metric tons of CO2 equivalent)',
        'Alternative and nuclear energy (% of total energy use)',
        'Renewable electricity output (% of total electricity output)',
        'Combustible renewables and waste (% of total energy)',
        'Public private partnerships investment in energy (current US$)'
    ]))
]

# Set 'Country Name' and 'Indicator Name' as index
correlation_data.set_index(['Country Name', 'Indicator Name'], inplace=True)

# Convert data to numeric (excluding non-numeric values)
correlation_data_numeric = correlation_data.apply(pd.to_numeric, errors='coerce')

# Create a correlation matrix
correlation_matrix = correlation_data_numeric.transpose().corr()

# Generate a heatmap with a different color map (viridis)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap showing Alternate Energy Used in India')
plt.show()
