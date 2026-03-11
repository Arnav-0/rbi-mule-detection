# Data Download Instructions

Place the following files in `data/raw/`:

## Transaction Files (split CSV)
- `transactions_part_0.csv` through `transactions_part_5.csv`

## Static Tables
- `customers.csv`
- `accounts.csv`
- `customer_account_linkage.csv`
- `product_details.csv`
- `train_labels.csv`
- `test_accounts.csv`

## Column Schema

### Transactions
| Column | Type | Description |
|--------|------|-------------|
| account_id | str | Account identifier |
| transaction_id | str | Unique transaction ID |
| transaction_date | datetime | Transaction timestamp |
| transaction_amount | float32 | Transaction amount (INR) |
| transaction_type | category | NEFT/UPI/IMPS/etc |
| channel | category | branch/online/mobile |
| counterparty_id | str | Counterparty account ID |
| is_credit | int8 | 1=credit, 0=debit |
| balance_after | float32 | Account balance after txn |

### Labels (train_labels.csv)
| Column | Type | Description |
|--------|------|-------------|
| account_id | str | Account identifier |
| is_mule | int | 1=mule, 0=legitimate |
