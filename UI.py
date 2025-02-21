import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

def load_q_table():
    try:
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table Loaded: Shape = {q_table.shape}")
        print("Q-table Sample:\n", q_table[:5, :5])
        return q_table
    except FileNotFoundError:
        print("Error: Q-table file not found.")
        return None
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

def create_price_mapping():
    possible_prices = np.arange(10, 20.5, 0.5)  # Prices: 10.0, 10.5, ... ,20.0 (21 values)
    mapping = {round(price, 2): i for i, price in enumerate(possible_prices)}
    return mapping, possible_prices

price_to_index, possible_prices = create_price_mapping()
print("🔍 Price-to-Index Mapping:", price_to_index)


def predict_price(state, q_table):
    try:
        state = float(state)
        # Use the nearest price in the mapping if there's no exact match.
        if state not in price_to_index:
            nearest_price = min(price_to_index.keys(), key=lambda x: abs(x - state))
            print(f"Warning: {state} not in price_to_index. Using nearest value: {nearest_price}")
            state = nearest_price
        state_idx = price_to_index[state]
        print(f"Input Price: {state} -> State Index: {state_idx}")
        
        # Get the best action (i.e. the column index with maximum Q-value for this state)
        action_idx = np.argmax(q_table[state_idx])
        if 0 <= action_idx < len(possible_prices):
            predicted_price = possible_prices[action_idx]
        else:
            predicted_price = "Invalid Prediction"
        
        print(f"Predicted Price: {predicted_price}")
        return predicted_price
    except Exception as e:
        print(f"Error in predict_price: {e}")
        return "No Prediction"


def create_ui():
    Q = load_q_table()
    if Q is None:
        return

    root = tk.Tk()
    root.title("Q-Learning Price Predictor")
    root.geometry("400x300")
    root.configure(bg="#f0f4f8")
    
    header = tk.Label(root, text="Price Predictor", font=("Helvetica", 20, "bold"),
                      bg="#4a90e2", fg="white", pady=10)
    header.pack(fill="x")
    
    main_frame = tk.Frame(root, bg="#f0f4f8")
    main_frame.pack(expand=True)
    
    tk.Label(main_frame, text="Enter Price:", font=("Helvetica", 12), bg="#f0f4f8")\
        .grid(row=0, column=0, padx=10, pady=10)
    
    input_entry = tk.Entry(main_frame, font=("Helvetica", 12), width=10, justify="center", bg="white")
    input_entry.grid(row=0, column=1, padx=10, pady=10)
    input_entry.insert(0, "10.00")  # Default starting price
    
    def on_enter(e):
        predict_button.config(bg="#357abd")
    
    def on_leave(e):
        predict_button.config(bg="#4a90e2")
    
    predict_button = tk.Button(main_frame, text="Predict", font=("Helvetica", 12, "bold"),
                               bg="#4a90e2", fg="white", relief="flat", padx=20, pady=5,
                               command=lambda: predict(input_entry, output_label, Q))
    predict_button.grid(row=1, column=0, columnspan=2, pady=20)
    predict_button.bind("<Enter>", on_enter)
    predict_button.bind("<Leave>", on_leave)
    
    output_label = tk.Label(main_frame, text="Predicted Price: ", font=("Helvetica", 14),
                             bg="#f0f4f8", fg="#333333")
    output_label.grid(row=2, column=0, columnspan=2, pady=10)
    
    def predict(entry, label, q_table):
        try:
            state = float(entry.get())
            predicted_price = predict_price(state, q_table)
            label.config(text=f"Predicted Price: {predicted_price}", fg="#2ecc71")
            predict_button.config(bg="#2ecc71")
            root.after(200, lambda: predict_button.config(bg="#4a90e2"))
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid price.")
            label.config(text="Predicted Price: ", fg="#333333")
    
    root.mainloop()


if __name__ == "__main__":
    create_ui()

    Q = load_q_table()
    if Q is not None:
        test_prices = [15.78, 15.26, 15.71, 16.23, 16.67, 16.54]
        print("\n Testing with example input prices:")
        for test_price in test_prices:
            predicted_price = predict_price(test_price, Q)
            print(f"Input Price: ${test_price:.2f} → Predicted Price: {predicted_price}")
