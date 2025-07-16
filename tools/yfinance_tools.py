import yfinance as yf
import logging
from datetime import date, timedelta

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tool Function ---
def get_weekly_stock_data(ticker: str):
    """
    A tool to fetch the last 7 days of historical daily data for a stock using yfinance.

    Args:
        ticker (str): The stock ticker symbol. For Indian stocks, it should end with '.NS'.
                      For example: "RELIANCE.NS", "HDFCBANK.NS".

    Returns:
        list: A list of dictionaries, where each dictionary represents a day's OHLCV data.
              Returns an error dictionary if the API call fails or the ticker is invalid.
    """
    try:
        # Calculate the date range for the last 7 calendar days.
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        logging.info(f"Fetching data for ticker: {ticker} from {start_date} to {end_date}")

        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Fetch historical data
        hist_df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if hist_df.empty:
            logging.warning(f"No data found for ticker: {ticker}. It may be an invalid ticker.")
            return {"status": "error", "message": f"No data found for ticker: {ticker}."}

        # Convert the DataFrame to a list of dictionaries
        hist_df = hist_df.reset_index()
        hist_df['Date'] = hist_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Rename columns to be more generic if needed (e.g., 'Close' -> 'close')
        hist_df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        logging.info(f"Data fetched successfully for ticker: {ticker}.")
        return hist_df[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

    except Exception as e:
        logging.error(f"An exception occurred while fetching data for {ticker}: {e}")
        return {"status": "error", "message": f"An exception occurred: {e}"}

# --- Testing Block ---
if __name__ == "__main__":
    logging.info("\n--- Running Test for yfinance_tools.py ---")
    
    # Example: Fetching data for HDFC Bank
    test_ticker = "HDFCBANK.NS"
    hdfc_data = get_weekly_stock_data(ticker=test_ticker)
    
    if isinstance(hdfc_data, list) and hdfc_data:
        print(f"\nSuccessfully fetched data for {test_ticker}:")
        # Print the data for the most recent day available.
        print(f"Most recent day's data: {hdfc_data[-1]}")
    elif isinstance(hdfc_data, dict) and hdfc_data.get('status') == 'error':
        print(f"\nError fetching data: {hdfc_data['message']}")
    else:
        print("\nNo data returned or an unknown issue occurred.")

