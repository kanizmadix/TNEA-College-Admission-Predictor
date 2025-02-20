import csv

def clean_and_save_csv(input_file, output_file):
    """
    Cleans the input CSV file, handling encoding issues and removing empty rows.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
    """
    try:
        # Try reading the CSV file with different encodings
        for encoding in ['utf-8', 'latin1', 'utf-16']:
            try:
                with open(input_file, 'r', encoding=encoding) as infile, \
                     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)

                    header = next(reader, None)
                    if header:
                        writer.writerow([col.strip() for col in header])  # Write header (stripped)
                        print(f"Header: {header}")

                    row_count = 0  # Debug: Keep track of written rows

                    for row in reader:
                        # Remove leading/trailing whitespace from each value in the row
                        cleaned_row = [value.strip() for value in row]

                        # Skip the row if it's completely empty after cleaning (all values are empty)
                        if all(not value for value in cleaned_row):
                            print(f"Skipping empty row: {row}")  # Debug: Log empty rows
                            continue

                        writer.writerow(cleaned_row)
                        row_count += 1

                    print(f"Number of rows written: {row_count}")  # Debug: Display row count
                    print(f"Cleaned data saved to {output_file}")
                break
            except UnicodeDecodeError:
                print(f"Failed to read {input_file} using encoding: {encoding}")
        else:
            print(f"Error: Could not decode file '{input_file}' with common encodings.")
            return


    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# File paths
vocational_input_csv_file = r"C:\Users\Kaniz\Pictures\TNEA_fct\Vocational_2023_Mark_Cutoff.csv"
vocational_output_csv_file = "cleaned_vocational_data.csv"

# Clean the Vocational CSV
clean_and_save_csv(vocational_input_csv_file, vocational_output_csv_file)