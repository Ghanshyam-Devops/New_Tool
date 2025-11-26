import streamlit as st
import streamlit.components.v1 as components
from auth import require_login

require_login()
def main():
    st.set_page_config(page_title="C5i Analytics Suite", layout="wide", page_icon="❄️")
    
    # Custom CSS
    st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 2.5em;
                color: var(--text-color, #29B5E8);
                margin-bottom: 20px;
                font-weight: 700;
            }
            .feature-card {
                height: 400px;
                padding: 20px;
                border-radius: 15px;
                background: var(--card-bg, #FFFFFF);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
                min-height: 300px;
                text-align: center;
            }
            .feature-card:hover {
                transform: translateY(-5px);
            }
            .feature-icon {
                font-size: 3.5em;
                margin-bottom: 15px;
                color: var(--icon-color, #29B5E8);
            }
            .feature-title {
                font-size: 1.6em;
                color: var(--text-color, #1E3F66);
                margin-bottom: 15px;
                font-weight: 600;
            }
            .feature-list {
                list-style-type: none;
                padding-left: 0;
                color: var(--text-color, #4A4A4A);
                line-height: 1.6;
            }
            .feature-list li {
                margin-bottom: 8px;
                padding-left: 1em;
                text-indent: -1em;
            }
            .feature-list li:before {
                content: "•";
                color: var(--icon-color, #29B5E8);
                display: inline-block;
                width: 1em;
                margin-left: -1em;
            }
            .user-guide-button {
                background: linear-gradient(90deg, #29B5E8, #1E3F66);
                color: white;
                font-size: 1.2em;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                display: block;
                margin: 20px auto;
                width: 250px;
                text-align: center;
            }
            .user-guide-button:hover {
                background: linear-gradient(90deg, #1E3F66, #29B5E8);
            }
            [data-testid=stSidebarContent] {
                background-color: #AEC0DA;

            }
            [data-testid=stSidebarNavItems] {
                font-size: 18px;
                background-color: #ffffff;
                padding: 20px;
                margin-top: 40px; /* Adjust this value as needed */
            }

        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://www.c5i.ai/wp-content/themes/course5iTheme/new-assets/images/c5i-primary-logo.svg' width='120'>
            <h1 class='title'>PROD Optim/Sim tools</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # # Core Features Grid
    # st.markdown("<h2 style='text-align: center; margin: 40px 0; color: var(--text-color, #1E3F66);'>PROD Optim/Sim tools</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("📈", "Scenario Simulation Engine", ["Run test scenarios with specific budget allocations across tactics to understand ROI and Revenue impacts                                   "]),
        ("⚙️", "Budget Optimizer", ["Run optimizations with a set budget or target Revenue to understand recommended budget allocation across tactics and its resulting KPI performance"]),
        ("🔍", "Report Analyzer", ["Compare optimizations ran with each other to understand overall impacts to KPI’s"])
    ]
    
        # Add custom CSS for fixed height
    

    # Render the features inside fixed-height cards
    for col, (icon, title, details) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3 class="feature-title">{title}</h3>
                    <ul class="feature-list">
                        {''.join(f'<li>{d}</li>' for d in details)}
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
    # User Guide Button
    
    # if st.button("📖 User Guide", key="user_guide", help="Click to open the User Guide", use_container_width=True):
    #     # with st.expander("Sonic Optimization / Simulation User Guide", expanded=True):
    #         st.warning("No File added yet")

if __name__ == "__main__":
    main()
