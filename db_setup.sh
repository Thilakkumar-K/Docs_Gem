#!/bin/bash

# Database Setup Script for RAG Document QA
# This script creates the PostgreSQL database and user

echo "🚀 Setting up PostgreSQL database for RAG Document QA..."

# Database configuration (modify these as needed)
DB_NAME="rag_document_qa"
DB_USER="postgres"
DB_PASSWORD="Password"  # Using the password from your .env file
DB_HOST="localhost"
DB_PORT="5432"

echo "📋 Database Configuration:"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo ""

# Function to execute SQL commands
execute_sql() {
    local sql_command="$1"
    echo "🔄 Executing: $sql_command"
    
    # Try with password from environment or prompt
    if [ -n "$PGPASSWORD" ]; then
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "$sql_command"
    else
        echo "💡 You may be prompted for the PostgreSQL password..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "$sql_command"
    fi
}

# Check if PostgreSQL is running
echo "🔍 Checking PostgreSQL connection..."
if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT 1;" >/dev/null 2>&1; then
    echo "❌ Cannot connect to PostgreSQL server."
    echo "   Please ensure:"
    echo "   1. PostgreSQL is installed and running"
    echo "   2. User '$DB_USER' exists and has appropriate permissions"
    echo "   3. Password is correct"
    echo "   4. Server is accepting connections on $DB_HOST:$DB_PORT"
    echo ""
    echo "   To install PostgreSQL on Ubuntu/Debian:"
    echo "   sudo apt update && sudo apt install postgresql postgresql-contrib"
    echo ""
    echo "   To start PostgreSQL:"
    echo "   sudo systemctl start postgresql"
    echo "   sudo systemctl enable postgresql"
    echo ""
    echo "   To create a user (run as postgres user):"
    echo "   sudo -u postgres createuser --interactive --pwprompt $DB_USER"
    exit 1
fi

echo "✅ PostgreSQL connection successful!"

# Check if database already exists
echo "🔍 Checking if database '$DB_NAME' exists..."
DB_EXISTS=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';")

if [ "$DB_EXISTS" = "1" ]; then
    echo "ℹ️ Database '$DB_NAME' already exists."
    read -p "❓ Do you want to recreate it? This will delete all existing data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️ Dropping existing database..."
        execute_sql "DROP DATABASE IF EXISTS $DB_NAME;"
    else
        echo "✅ Using existing database."
        exit 0
    fi
fi

# Create the database
echo "🏗️ Creating database '$DB_NAME'..."
execute_sql "CREATE DATABASE $DB_NAME WITH ENCODING='UTF8' LC_COLLATE='en_US.UTF-8' LC_CTYPE='en_US.UTF-8';"

if [ $? -eq 0 ]; then
    echo "✅ Database '$DB_NAME' created successfully!"
else
    echo "❌ Failed to create database. Please check the error messages above."
    exit 1
fi

# Grant permissions (if needed)
echo "🔐 Setting up permissions..."
execute_sql "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Test connection to the new database
echo "🧪 Testing connection to new database..."
if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 'Database connection successful!' as message;" >/dev/null 2>&1; then
    echo "✅ Successfully connected to database '$DB_NAME'!"
else
    echo "❌ Failed to connect to the new database."
    exit 1
fi

echo ""
echo "🎉 Database setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "   1. Run: python init_db.py"
echo "   2. Start the application: python main.py"
echo ""
echo "🔗 Connection details:"
echo "   Database URL: postgresql+asyncpg://$DB_USER:***@$DB_HOST:$DB_PORT/$DB_NAME"
echo "   Direct psql: psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"