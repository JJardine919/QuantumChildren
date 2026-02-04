
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sqlite3
from datetime import datetime
import pandas as pd
from tkcalendar import DateEntry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Dictionary for translations
TRANSLATIONS = {
    'ru': {
        'title': 'Домашняя бухгалтерия',
        'transactions': 'Транзакции',
        'assets': 'Активы/Пассивы',
        'reports': 'Отчеты',
        'new_transaction': 'Новая транзакция',
        'date': 'Дата',
        'type': 'Тип',
        'category': 'Категория',
        'amount': 'Сумма',
        'description': 'Описание',
        'add': 'Добавить',
        'delete': 'Удалить выбранное',
        'income': 'Доход',
        'expense': 'Расход',
        'new_asset': 'Новый актив/пассив',
        'name': 'Наименование',
        'asset': 'Актив',
        'liability': 'Пассив',
        'value': 'Стоимость',
        'report_period': 'Период отчета',
        'from': 'С',
        'to': 'По',
        'generate_report': 'Сформировать отчет',
        'text_report': 'Текстовый отчет',
        'balance_chart': 'График баланса',
        'success': 'Успех',
        'error': 'Ошибка',
        'warning': 'Предупреждение',
        'confirm': 'Подтверждение',
        'transaction_added': 'Транзакция добавлена успешно!',
        'check_data': 'Проверьте правильность введенных данных!',
        'asset_added': 'Актив/пассив добавлен успешно!',
        'select_transaction': 'Выберите транзакцию для удаления!',
        'delete_transaction_confirm': 'Удалить выбранную транзакцию?',
        'transaction_deleted': 'Транзакция удалена из базы данных',
        'select_asset': 'Выберите актив/пассив для удаления!',
        'delete_asset_confirm': 'Удалить выбранный актив/пассив?',
        'asset_deleted': 'Актив/пассив удален из базы данных',
        'cash_flow': 'ДВИЖЕНИЕ ДЕНЕЖНЫХ СРЕДСТВ',
        'total_balance': 'Общий баланс',
        'assets_liabilities_balance': 'БАЛАНС АКТИВОВ И ПАССИВОВ',
        'net_worth': 'Чистая стоимость',
        'balance_dynamics': 'Динамика баланса',
        'balance': 'Баланс',
        'switch_language': 'Switch to English',
        'assets_liabilities_chart': 'График баланса активов и пассивов',
        'total_balance_chart': 'Общий баланс остатка всех средств'
    },
    'en': {
        'title': 'Home Accounting',
        'transactions': 'Transactions',
        'assets': 'Assets/Liabilities',
        'reports': 'Reports',
        'new_transaction': 'New Transaction',
        'date': 'Date',
        'type': 'Type',
        'category': 'Category',
        'amount': 'Amount',
        'description': 'Description',
        'add': 'Add',
        'delete': 'Delete Selected',
        'income': 'Income',
        'expense': 'Expense',
        'new_asset': 'New Asset/Liability',
        'name': 'Name',
        'asset': 'Asset',
        'liability': 'Liability',
        'value': 'Value',
        'report_period': 'Report Period',
        'from': 'From',
        'to': 'To',
        'generate_report': 'Generate Report',
        'text_report': 'Text Report',
        'balance_chart': 'Balance Chart',
        'success': 'Success',
        'error': 'Error',
        'warning': 'Warning',
        'confirm': 'Confirmation',
        'transaction_added': 'Transaction added successfully!',
        'check_data': 'Please check the entered data!',
        'asset_added': 'Asset/Liability added successfully!',
        'select_transaction': 'Please select a transaction to delete!',
        'delete_transaction_confirm': 'Delete selected transaction?',
        'transaction_deleted': 'Transaction deleted from database',
        'select_asset': 'Please select an asset/liability to delete!',
        'delete_asset_confirm': 'Delete selected asset/liability?',
        'asset_deleted': 'Asset/Liability deleted from database',
        'cash_flow': 'CASH FLOW',
        'total_balance': 'Total Balance',
        'assets_liabilities_balance': 'ASSETS AND LIABILITIES BALANCE',
        'net_worth': 'Net Worth',
        'balance_dynamics': 'Balance Dynamics',
        'balance': 'Balance',
        'switch_language': 'Переключить на русский',
        'assets_liabilities_chart': 'Assets and Liabilities Balance Chart',
        'total_balance_chart': 'Total Balance of All Remaining Funds'
    }
}

class HomeAccounting:
    def __init__(self, root):
        self.root = root
        self.current_language = 'ru'  # Default language
        self.setup_ui()

    def setup_ui(self):
        self.create_database()
        self.setup_main_window()
        self.create_language_button()
        self.setup_notebook()
        self.setup_transactions_tab()
        self.setup_assets_tab()
        self.setup_reports_tab()

    def setup_main_window(self):
        self.root.title(TRANSLATIONS[self.current_language]['title'])
        self.root.geometry("1200x700")

    def create_language_button(self):
        self.lang_button = ttk.Button(
            self.root,
            text=TRANSLATIONS[self.current_language]['switch_language'],
            command=self.toggle_language
        )
        self.lang_button.pack(anchor='ne', padx=10, pady=5)

    def toggle_language(self):
        self.current_language = 'en' if self.current_language == 'ru' else 'ru'
        self.update_ui_language()

    def update_ui_language(self):
        # Update main window title
        self.root.title(TRANSLATIONS[self.current_language]['title'])

        # Update language button
        self.lang_button.config(text=TRANSLATIONS[self.current_language]['switch_language'])

        # Update notebook tabs
        self.notebook.tab(self.transactions_tab, text=TRANSLATIONS[self.current_language]['transactions'])
        self.notebook.tab(self.assets_tab, text=TRANSLATIONS[self.current_language]['assets'])
        self.notebook.tab(self.reports_tab, text=TRANSLATIONS[self.current_language]['reports'])

        # Update frame titles
        self.transactions_input_frame.config(text=TRANSLATIONS[self.current_language]['new_transaction'])
        self.assets_input_frame.config(text=TRANSLATIONS[self.current_language]['new_asset'])
        self.period_frame.config(text=TRANSLATIONS[self.current_language]['report_period'])

        # Update buttons in transactions frame
        for widget in self.transactions_input_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                if widget.cget('text') in [TRANSLATIONS['ru']['add'], TRANSLATIONS['en']['add']]:
                    widget.config(text=TRANSLATIONS[self.current_language]['add'])

        # Update buttons in assets frame
        for widget in self.assets_input_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                if widget.cget('text') in [TRANSLATIONS['ru']['add'], TRANSLATIONS['en']['add']]:
                    widget.config(text=TRANSLATIONS[self.current_language]['add'])

        # Update delete buttons
        for widget in self.transactions_tab.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget('text') in [TRANSLATIONS['ru']['delete'], TRANSLATIONS['en']['delete']]:
                widget.config(text=TRANSLATIONS[self.current_language]['delete'])

        for widget in self.assets_tab.winfo_children():
            if isinstance(widget, ttk.Button) and widget.cget('text') in [TRANSLATIONS['ru']['delete'], TRANSLATIONS['en']['delete']]:
                widget.config(text=TRANSLATIONS[self.current_language]['delete'])

        # Update transactions tab
        self.update_transactions_tab_language()

        # Update assets tab
        self.update_assets_tab_language()

        # Update reports tab
        self.update_reports_tab_language()

        # Refresh data displays
        self.load_transactions()
        self.load_assets()

    def setup_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)

        self.transactions_tab = ttk.Frame(self.notebook)
        self.assets_tab = ttk.Frame(self.notebook)
        self.reports_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.transactions_tab, text=TRANSLATIONS[self.current_language]['transactions'])
        self.notebook.add(self.assets_tab, text=TRANSLATIONS[self.current_language]['assets'])
        self.notebook.add(self.reports_tab, text=TRANSLATIONS[self.current_language]['reports'])

    def update_transactions_tab_language(self):
        # Update transaction input frame
        for widget in self.transactions_input_frame.winfo_children():
            if isinstance(widget, ttk.Label):
                for key, value in TRANSLATIONS[self.current_language].items():
                    if widget.cget('text').lower() in [TRANSLATIONS['ru'][key].lower(), TRANSLATIONS['en'][key].lower()]:
                        widget.config(text=value)
                        break

        # Update transaction type combobox values
        self.type_combo['values'] = [
            TRANSLATIONS[self.current_language]['income'],
            TRANSLATIONS[self.current_language]['expense']
        ]

        # Update treeview headers
        self.transactions_tree.heading("date", text=TRANSLATIONS[self.current_language]['date'])
        self.transactions_tree.heading("type", text=TRANSLATIONS[self.current_language]['type'])
        self.transactions_tree.heading("category", text=TRANSLATIONS[self.current_language]['category'])
        self.transactions_tree.heading("amount", text=TRANSLATIONS[self.current_language]['amount'])
        self.transactions_tree.heading("description", text=TRANSLATIONS[self.current_language]['description'])

    def update_assets_tab_language(self):
        # Update asset input frame
        for widget in self.assets_input_frame.winfo_children():
            if isinstance(widget, ttk.Label):
                for key, value in TRANSLATIONS[self.current_language].items():
                    if widget.cget('text').lower() in [TRANSLATIONS['ru'][key].lower(), TRANSLATIONS['en'][key].lower()]:
                        widget.config(text=value)
                        break

        # Update asset type combobox values
        self.asset_type_combo['values'] = [
            TRANSLATIONS[self.current_language]['asset'],
            TRANSLATIONS[self.current_language]['liability']
        ]

        # Update treeview headers
        self.assets_tree.heading("name", text=TRANSLATIONS[self.current_language]['name'])
        self.assets_tree.heading("type", text=TRANSLATIONS[self.current_language]['type'])
        self.assets_tree.heading("value", text=TRANSLATIONS[self.current_language]['value'])
        self.assets_tree.heading("date", text=TRANSLATIONS[self.current_language]['date'])

    def update_reports_tab_language(self):
        # Update report period frame
        self.period_frame.config(text=TRANSLATIONS[self.current_language]['report_period'])

        # Update report labels and buttons
        for widget in self.period_frame.winfo_children():
            if isinstance(widget, ttk.Label):
                if widget.cget('text') == TRANSLATIONS['ru']['from'] or widget.cget('text') == TRANSLATIONS['en']['from']:
                    widget.config(text=TRANSLATIONS[self.current_language]['from'])
                elif widget.cget('text') == TRANSLATIONS['ru']['to'] or widget.cget('text') == TRANSLATIONS['en']['to']:
                    widget.config(text=TRANSLATIONS[self.current_language]['to'])
            elif isinstance(widget, ttk.Button):
                if widget.cget('text') in [TRANSLATIONS['ru']['generate_report'], TRANSLATIONS['en']['generate_report']]:
                    widget.config(text=TRANSLATIONS[self.current_language]['generate_report'])

    def create_database(self):
        conn = sqlite3.connect('home_accounting.db')
        c = conn.cursor()

        # Creating transactions table
        c.execute('''CREATE TABLE IF NOT EXISTS transactions
                    (id INTEGER PRIMARY KEY,
                     date TEXT,
                     type TEXT,
                     category TEXT,
                     amount REAL,
                     description TEXT)''')

        # Creating assets table
        c.execute('''CREATE TABLE IF NOT EXISTS assets
                    (id INTEGER PRIMARY KEY,
                     name TEXT,
                     type TEXT,
                     value REAL,
                     date TEXT)''')

        conn.commit()
        conn.close()

    def setup_transactions_tab(self):
        # Transaction input frame
        self.transactions_input_frame = ttk.LabelFrame(
            self.transactions_tab,
            text=TRANSLATIONS[self.current_language]['new_transaction'],
            padding=10
        )
        self.transactions_input_frame.pack(fill='x', padx=5, pady=5)

        # Date
        ttk.Label(self.transactions_input_frame, text=TRANSLATIONS[self.current_language]['date']).grid(
            row=0, column=0, padx=5, pady=5)
        self.date_entry = DateEntry(self.transactions_input_frame, width=12, background='darkblue',
                                  foreground='white', borderwidth=2)
        self.date_entry.grid(row=0, column=1, padx=5, pady=5)

        # Type
        ttk.Label(self.transactions_input_frame, text=TRANSLATIONS[self.current_language]['type']).grid(
            row=0, column=2, padx=5, pady=5)
        self.type_combo = ttk.Combobox(self.transactions_input_frame,
            values=[TRANSLATIONS[self.current_language]['income'],
                   TRANSLATIONS[self.current_language]['expense']])
        self.type_combo.grid(row=0, column=3, padx=5, pady=5)
        self.type_combo.set(TRANSLATIONS[self.current_language]['income'])

        # Category
        ttk.Label(self.transactions_input_frame, text=TRANSLATIONS[self.current_language]['category']).grid(
            row=0, column=4, padx=5, pady=5)
        self.category_entry = ttk.Entry(self.transactions_input_frame)
        self.category_entry.grid(row=0, column=5, padx=5, pady=5)

        # Amount
        ttk.Label(self.transactions_input_frame, text=TRANSLATIONS[self.current_language]['amount']).grid(
            row=1, column=0, padx=5, pady=5)
        self.amount_entry = ttk.Entry(self.transactions_input_frame)
        self.amount_entry.grid(row=1, column=1, padx=5, pady=5)

        # Description
        ttk.Label(self.transactions_input_frame, text=TRANSLATIONS[self.current_language]['description']).grid(
            row=1, column=2, padx=5, pady=5)
        self.description_entry = ttk.Entry(self.transactions_input_frame, width=40)
        self.description_entry.grid(row=1, column=3, columnspan=3, padx=5, pady=5)

        # Add button
        ttk.Button(self.transactions_input_frame,
                  text=TRANSLATIONS[self.current_language]['add'],
                  command=self.add_transaction).grid(row=2, column=0, columnspan=6, pady=10)

        # Transactions tree
        self.transactions_tree = ttk.Treeview(self.transactions_tab,
                                            columns=("date", "type", "category", "amount", "description"),
                                            show='headings')

        self.transactions_tree.heading("date", text=TRANSLATIONS[self.current_language]['date'])
        self.transactions_tree.heading("type", text=TRANSLATIONS[self.current_language]['type'])
        self.transactions_tree.heading("category", text=TRANSLATIONS[self.current_language]['category'])
        self.transactions_tree.heading("amount", text=TRANSLATIONS[self.current_language]['amount'])
        self.transactions_tree.heading("description", text=TRANSLATIONS[self.current_language]['description'])

        self.transactions_tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Delete button
        ttk.Button(self.transactions_tab,
                  text=TRANSLATIONS[self.current_language]['delete'],
                  command=self.delete_transaction).pack(pady=5)

        self.load_transactions()

    def setup_assets_tab(self):
        # Asset input frame
        self.assets_input_frame = ttk.LabelFrame(
            self.assets_tab,
            text=TRANSLATIONS[self.current_language]['new_asset'],
            padding=10
        )
        self.assets_input_frame.pack(fill='x', padx=5, pady=5)

        # Name
        ttk.Label(self.assets_input_frame, text=TRANSLATIONS[self.current_language]['name']).grid(
            row=0, column=0, padx=5, pady=5)
        self.asset_name_entry = ttk.Entry(self.assets_input_frame)
        self.asset_name_entry.grid(row=0, column=1, padx=5, pady=5)

        # Type
        ttk.Label(self.assets_input_frame, text=TRANSLATIONS[self.current_language]['type']).grid(
            row=0, column=2, padx=5, pady=5)
        self.asset_type_combo = ttk.Combobox(self.assets_input_frame,
            values=[TRANSLATIONS[self.current_language]['asset'],
                   TRANSLATIONS[self.current_language]['liability']])
        self.asset_type_combo.grid(row=0, column=3, padx=5, pady=5)
        self.asset_type_combo.set(TRANSLATIONS[self.current_language]['asset'])

        # Value
        ttk.Label(self.assets_input_frame, text=TRANSLATIONS[self.current_language]['value']).grid(
            row=0, column=4, padx=5, pady=5)
        self.asset_value_entry = ttk.Entry(self.assets_input_frame)
        self.asset_value_entry.grid(row=0, column=5, padx=5, pady=5)

        # Add button
        ttk.Button(self.assets_input_frame,
                  text=TRANSLATIONS[self.current_language]['add'],
                  command=self.add_asset).grid(row=1, column=0, columnspan=6, pady=10)

        # Assets tree
        self.assets_tree = ttk.Treeview(self.assets_tab,
                                      columns=("name", "type", "value", "date"),
                                      show='headings')

        self.assets_tree.heading("name", text=TRANSLATIONS[self.current_language]['name'])
        self.assets_tree.heading("type", text=TRANSLATIONS[self.current_language]['type'])
        self.assets_tree.heading("value", text=TRANSLATIONS[self.current_language]['value'])
        self.assets_tree.heading("date", text=TRANSLATIONS[self.current_language]['date'])

        self.assets_tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Delete button
        ttk.Button(self.assets_tab,
                  text=TRANSLATIONS[self.current_language]['delete'],
                  command=self.delete_asset).pack(pady=5)

        self.load_assets()

    def setup_reports_tab(self):
        # Period frame
        self.period_frame = ttk.LabelFrame(
            self.reports_tab,
            text=TRANSLATIONS[self.current_language]['report_period'],
            padding=10
        )
        self.period_frame.pack(fill='x', padx=5, pady=5)

        # Report notebook
        self.report_notebook = ttk.Notebook(self.reports_tab)
        self.report_notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Text report and graph tabs
        self.text_report_tab = ttk.Frame(self.report_notebook)
        self.graph_report_tab = ttk.Frame(self.report_notebook)

        self.report_notebook.add(self.text_report_tab,
                               text=TRANSLATIONS[self.current_language]['text_report'])
        self.report_notebook.add(self.graph_report_tab,
                               text=TRANSLATIONS[self.current_language]['balance_chart'])

        # Date range
        ttk.Label(self.period_frame, text=TRANSLATIONS[self.current_language]['from']).grid(
            row=0, column=0, padx=5, pady=5)
        self.start_date = DateEntry(self.period_frame, width=12, background='darkblue',
                                  foreground='white', borderwidth=2)
        self.start_date.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.period_frame, text=TRANSLATIONS[self.current_language]['to']).grid(
            row=0, column=2, padx=5, pady=5)
        self.end_date = DateEntry(self.period_frame, width=12, background='darkblue',
                                foreground='white', borderwidth=2)
        self.end_date.grid(row=0, column=3, padx=5, pady=5)

        # Generate report button
        ttk.Button(self.period_frame,
                  text=TRANSLATIONS[self.current_language]['generate_report'],
                  command=self.generate_report).grid(row=0, column=4, padx=20, pady=5)

        # Text report area
        self.report_text = tk.Text(self.text_report_tab, height=20, width=70)
        self.report_text.pack(padx=5, pady=5)

        # Graph area
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.graph_report_tab)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def add_transaction(self):
        try:
            date = self.date_entry.get_date().strftime("%Y-%m-%d")
            type_ = self.type_combo.get()
            # Convert type to Russian for storage if currently in English
            if self.current_language == 'en':
                type_ = self.translate_type(type_, reverse=True)
            category = self.category_entry.get()
            amount = float(self.amount_entry.get())
            description = self.description_entry.get()

            conn = sqlite3.connect('home_accounting.db')
            c = conn.cursor()
            c.execute('''INSERT INTO transactions (date, type, category, amount, description)
                        VALUES (?, ?, ?, ?, ?)''',
                     (date, type_, category, amount, description))
            conn.commit()
            conn.close()

            self.load_transactions()
            self.clear_transaction_entries()
            messagebox.showinfo(
                TRANSLATIONS[self.current_language]['success'],
                TRANSLATIONS[self.current_language]['transaction_added']
            )

        except ValueError:
            messagebox.showerror(
                TRANSLATIONS[self.current_language]['error'],
                TRANSLATIONS[self.current_language]['check_data']
            )

    def add_asset(self):
        try:
            name = self.asset_name_entry.get()
            type_ = self.asset_type_combo.get()
            # Convert type to Russian for storage if currently in English
            if self.current_language == 'en':
                type_ = self.translate_type(type_, reverse=True)
            value = float(self.asset_value_entry.get())
            date = datetime.now().strftime("%Y-%m-%d")

            conn = sqlite3.connect('home_accounting.db')
            c = conn.cursor()
            c.execute('''INSERT INTO assets (name, type, value, date)
                        VALUES (?, ?, ?, ?)''', (name, type_, value, date))
            conn.commit()
            conn.close()

            self.load_assets()
            self.clear_asset_entries()
            messagebox.showinfo(
                TRANSLATIONS[self.current_language]['success'],
                TRANSLATIONS[self.current_language]['asset_added']
            )

        except ValueError:
            messagebox.showerror(
                TRANSLATIONS[self.current_language]['error'],
                TRANSLATIONS[self.current_language]['check_data']
            )

    def translate_type(self, type_value, reverse=False):
        """Translate transaction and asset types between languages"""
        translations = {
            'Доход': 'Income',
            'Расход': 'Expense',
            'Актив': 'Asset',
            'Пассив': 'Liability'
        }

        if reverse:
            translations = {v: k for k, v in translations.items()}

        return translations.get(type_value, type_value)

    def load_transactions(self):
        for item in self.transactions_tree.get_children():
            self.transactions_tree.delete(item)

        conn = sqlite3.connect('home_accounting.db')
        c = conn.cursor()
        c.execute('''SELECT date, type, category, amount, description FROM transactions
                    ORDER BY date DESC''')

        for row in c.fetchall():
            # Convert row to list for modification
            row = list(row)
            # Translate type if needed
            if self.current_language == 'en':
                row[1] = self.translate_type(row[1])
            self.transactions_tree.insert('', 'end', values=row)

        conn.close()

    def load_assets(self):
        for item in self.assets_tree.get_children():
            self.assets_tree.delete(item)

        conn = sqlite3.connect('home_accounting.db')
        c = conn.cursor()
        c.execute('''SELECT name, type, value, date FROM assets ORDER BY date DESC''')

        for row in c.fetchall():
            # Convert row to list for modification
            row = list(row)
            # Translate type if needed
            if self.current_language == 'en':
                row[1] = self.translate_type(row[1])
            self.assets_tree.insert('', 'end', values=row)

        conn.close()

    def delete_transaction(self):
        selected_item = self.transactions_tree.selection()
        if not selected_item:
            messagebox.showwarning(
                TRANSLATIONS[self.current_language]['warning'],
                TRANSLATIONS[self.current_language]['select_transaction']
            )
            return

        if messagebox.askyesno(
            TRANSLATIONS[self.current_language]['confirm'],
            TRANSLATIONS[self.current_language]['delete_transaction_confirm']
        ):
            values = self.transactions_tree.item(selected_item)['values']
            conn = sqlite3.connect('home_accounting.db')
            c = conn.cursor()
            c.execute('''DELETE FROM transactions
                        WHERE date=? AND type=? AND category=? AND amount=? AND description=?''',
                     values)
            conn.commit()
            conn.close()

            self.transactions_tree.delete(selected_item)
            messagebox.showinfo(
                TRANSLATIONS[self.current_language]['success'],
                TRANSLATIONS[self.current_language]['transaction_deleted']
            )

    def delete_asset(self):
        selected_item = self.assets_tree.selection()
        if not selected_item:
            messagebox.showwarning(
                TRANSLATIONS[self.current_language]['warning'],
                TRANSLATIONS[self.current_language]['select_asset']
            )
            return

        if messagebox.askyesno(
            TRANSLATIONS[self.current_language]['confirm'],
            TRANSLATIONS[self.current_language]['delete_asset_confirm']
        ):
            values = self.assets_tree.item(selected_item)['values']
            conn = sqlite3.connect('home_accounting.db')
            c = conn.cursor()
            c.execute('''DELETE FROM assets
                        WHERE name=? AND type=? AND value=? AND date=?''',
                     values)
            conn.commit()
            conn.close()

            self.assets_tree.delete(selected_item)
            messagebox.showinfo(
                TRANSLATIONS[self.current_language]['success'],
                TRANSLATIONS[self.current_language]['asset_deleted']
            )

    def generate_report(self):
        start = self.start_date.get_date()
        end = self.end_date.get_date()

        # Format dates for SQL query
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        conn = sqlite3.connect('home_accounting.db')

        # Get transaction data
        transactions_df = pd.read_sql_query(f'''
            SELECT date, type, amount
            FROM transactions
            WHERE date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY date
        ''', conn)

        # Get assets and liabilities data
        assets_df = pd.read_sql_query(f'''
            SELECT type, SUM(value) as total
            FROM assets
            WHERE date <= '{end_str}'
            GROUP BY type
        ''', conn)

        conn.close()

        # Generate report
        report = f"{TRANSLATIONS[self.current_language]['report_period']}: {start_str} - {end_str}\n\n"

        # Analyze income and expenses
        if not transactions_df.empty:
            report += f"{TRANSLATIONS[self.current_language]['cash_flow']}\n"
            report += "-" * 30 + "\n"

            income = transactions_df[transactions_df['type'] == 'Доход']['amount'].sum()
            expenses = transactions_df[transactions_df['type'] == 'Расход']['amount'].sum()
            balance = income - expenses

            report += f"{TRANSLATIONS[self.current_language]['income']}: {income:,.2f}\n"
            report += f"{TRANSLATIONS[self.current_language]['expense']}: {expenses:,.2f}\n"
            report += "-" * 30 + "\n"
            report += f"{TRANSLATIONS[self.current_language]['total_balance']}: {balance:,.2f}\n\n"

        # Analyze assets and liabilities
        if not assets_df.empty:
            report += f"{TRANSLATIONS[self.current_language]['assets_liabilities_balance']}\n"
            report += "-" * 30 + "\n"

            assets_total = assets_df[assets_df['type'] == 'Актив']['total'].sum()
            liabilities_total = assets_df[assets_df['type'] == 'Пассив']['total'].sum()
            net_worth = assets_total - liabilities_total

            report += f"{TRANSLATIONS[self.current_language]['asset']}: {assets_total:,.2f}\n"
            report += f"{TRANSLATIONS[self.current_language]['liability']}: {liabilities_total:,.2f}\n"
            report += "-" * 30 + "\n"
            report += f"{TRANSLATIONS[self.current_language]['net_worth']}: {net_worth:,.2f}\n"

        # Update text report
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)

        # Generate balance charts
        self.plot_balance_chart(transactions_df)
        self.plot_assets_liabilities_chart(assets_df)
        self.plot_total_balance_chart(transactions_df, assets_df)

    def plot_balance_chart(self, transactions_df):
        if transactions_df.empty:
            return

        # Convert dates to datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])

        # Create date range
        date_range = pd.date_range(transactions_df['date'].min(),
                                 transactions_df['date'].max(),
                                 freq='D')

        # Calculate cumulative balance
        transactions_df['amount'] = transactions_df.apply(
            lambda x: x['amount'] if x['type'] == 'Доход' else -x['amount'],
            axis=1
        )
        daily_balance = transactions_df.groupby('date')['amount'].sum().reindex(date_range).fillna(0)
        cumulative_balance = daily_balance.cumsum()

        # Clear previous plot
        self.plot.clear()

        # Create new plot
        self.plot.plot(cumulative_balance.index, cumulative_balance.values,
                      marker='o', linestyle='-', linewidth=2, markersize=4)

        # Set labels and title
        self.plot.set_title(TRANSLATIONS[self.current_language]['balance_dynamics'])
        self.plot.set_xlabel(TRANSLATIONS[self.current_language]['date'])
        self.plot.set_ylabel(TRANSLATIONS[self.current_language]['balance'])
        self.plot.grid(True)

        # Format x-axis dates
        self.plot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.figure.autofmt_xdate()

        # Update canvas
        self.canvas.draw()

    def plot_assets_liabilities_chart(self, assets_df):
        if assets_df.empty:
            return

        # Clear previous plot
        self.plot.clear()

        # Create new plot
        assets_total = assets_df[assets_df['type'] == 'Актив']['total'].sum()
        liabilities_total = assets_df[assets_df['type'] == 'Пассив']['total'].sum()

        self.plot.bar(['Assets', 'Liabilities'], [assets_total, liabilities_total], color=['green', 'red'])

        # Set labels and title
        self.plot.set_title(TRANSLATIONS[self.current_language]['assets_liabilities_chart'])
        self.plot.set_xlabel(TRANSLATIONS[self.current_language]['type'])
        self.plot.set_ylabel(TRANSLATIONS[self.current_language]['value'])
        self.plot.grid(True)

        # Update canvas
        self.canvas.draw()

    def plot_total_balance_chart(self, transactions_df, assets_df):
        if transactions_df.empty or assets_df.empty:
            return

        # Convert dates to datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])

        # Create date range
        date_range = pd.date_range(transactions_df['date'].min(),
                                 transactions_df['date'].max(),
                                 freq='D')

        # Calculate cumulative balance
        transactions_df['amount'] = transactions_df.apply(
            lambda x: x['amount'] if x['type'] == 'Доход' else -x['amount'],
            axis=1
        )
        daily_balance = transactions_df.groupby('date')['amount'].sum().reindex(date_range).fillna(0)
        cumulative_balance = daily_balance.cumsum()

        # Calculate net worth
        assets_total = assets_df[assets_df['type'] == 'Актив']['total'].sum()
        liabilities_total = assets_df[assets_df['type'] == 'Пассив']['total'].sum()
        net_worth = assets_total - liabilities_total

        # Calculate total balance
        total_balance = cumulative_balance + net_worth

        # Clear previous plot
        self.plot.clear()

        # Create new plot
        self.plot.plot(total_balance.index, total_balance.values,
                      marker='o', linestyle='-', linewidth=2, markersize=4)

        # Set labels and title
        self.plot.set_title(TRANSLATIONS[self.current_language]['total_balance_chart'])
        self.plot.set_xlabel(TRANSLATIONS[self.current_language]['date'])
        self.plot.set_ylabel(TRANSLATIONS[self.current_language]['balance'])
        self.plot.grid(True)

        # Format x-axis dates
        self.plot.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.figure.autofmt_xdate()

        # Update canvas
        self.canvas.draw()

    def clear_transaction_entries(self):
        """Clear transaction input fields"""
        self.category_entry.delete(0, tk.END)
        self.amount_entry.delete(0, tk.END)
        self.description_entry.delete(0, tk.END)
        self.type_combo.set(TRANSLATIONS[self.current_language]['income'])

    def clear_asset_entries(self):
        """Clear asset input fields"""
        self.asset_name_entry.delete(0, tk.END)
        self.asset_value_entry.delete(0, tk.END)
        self.asset_type_combo.set(TRANSLATIONS[self.current_language]['asset'])

if __name__ == "__main__":
    root = tk.Tk()
    app = HomeAccounting(root)

    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')

    style.configure("Treeview",
                   background="#F0F0F0",
                   foreground="black",
                   rowheight=25,
                   fieldbackground="#F0F0F0")

    style.configure("Treeview.Heading",
                   background="#E1E1E1",
                   foreground="black",
                   relief="flat")

    style.map("Treeview",
              background=[("selected", "#0078D7")])

    root.mainloop()
