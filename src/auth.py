# src/auth.py
import streamlit as st
from config import APP_AUTH_KEY


def require_login():
    """
    Simple password-gate. Renders a login screen
    and blocks the rest of the app until you enter
    the correct APP_AUTH_KEY.
    """
    # Initialize
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If not yet authenticated, show login form
    if not st.session_state.authenticated:
        st.title("🔐 Please log in")
        key = st.text_input("Enter access key:", type="password")
        if st.button("Login"):
            if key and key == APP_AUTH_KEY:
                st.session_state.authenticated = True
                st.success("Welcome! 🎉")

                # Try to rerun; if that's unavailable, ask for manual refresh
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
                else:
                    st.info("Please refresh the page to continue.")
                    st.stop()
            else:
                st.error("❌ Invalid key, try again.")
        # Prevent the rest of the script from running
        st.stop()
