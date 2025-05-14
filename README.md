# Football Match Analysis UI

A comprehensive football match analysis and visualization application built with Streamlit.

## Features

- Interactive match selection and organization
- Multiple visualization types:
  - Pitch visualization
  - Scatter plots
  - Bar charts
  - Radar charts
  - Pizza charts
  - Data tables
- Advanced filtering and data analysis
- Player and team performance metrics
- Customizable visualizations
- Data export capabilities

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run msgpack_test.py
```

## Project Structure

- `msgpack_test.py`: Main application entry point
- `pitch_tab.py`: Football pitch visualization
- `scatter_tab.py`: Scatter plot visualization
- `table_tab.py`: Data table visualization
- `bar_tab.py`: Bar chart visualization
- `radar_tab.py`: Radar chart visualization
- `pizza_tab.py`: Pie chart visualization
- `src/`: Source code directory
  - `visualization/`: Visualization components
  - `utils/`: Utility functions
  - `components/`: Reusable UI components
  - `data/`: Data handling

## Dependencies

- Streamlit
- Matplotlib
- Pandas
- NumPy
- mplsoccer
- msgpack
- PIL
- requests

## Usage

1. Launch the application
2. Select matches using the match selection interface
3. Choose visualization type
4. Customize the visualization using the provided options
5. Export or share your analysis

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 