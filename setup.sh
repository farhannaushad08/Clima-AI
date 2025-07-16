mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
enableXsrfProtection = false\n\
\n\
" > ~/.streamlit/config.toml
