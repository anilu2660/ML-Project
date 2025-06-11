root = tk.Tk()
# root.title("House Price Prediction")

# labels = [
#     "Bedrooms", "Bathrooms", "Floors", "Condition", "View",
#     "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
#     "Street (label)", "City (label)", "StateZip (label)", "Country (label)", "Year Built"
# ]

# entries = []

# for idx, label in enumerate(labels):
#     tk.Label(root, text=label).grid(row=idx, column=0)
#     entry = tk.Entry(root)
#     entry.grid(row=idx, column=1)
#     entries.append(entry)

# (bedroom_entry, bathroom_entry, floor_entry, condition_entry, view_entry,
#  sqft_living_entry, sqft_lot_entry, sqft_above_entry, sqft_basement_entry,
#  street_entry, city_entry, statezip_entry, country_entry, year_built_entry) = entries

# # Prediction Button
# predict_button = tk.Button(root, text="Predict Price", command=predict_price)
# predict_button.grid(row=len(labels), column=0, columnspan=2)
