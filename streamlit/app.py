import streamlit as st
import torch
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# --- Model Metadata ---
MODEL_METADATA = {
    'pericardial-effusion': {
        'type': 'binary_classification',
        'classes': ['Absent', 'Present'],
        'description': 'Pericardial effusion',
        'unit': ''
    },
    'EF': {
        'type': 'regression',
        'description': 'Left ventricular ejection fraction',
        'unit': '%',
        'normal_range': (55, 70)
    },
    'GLS': {
        'type': 'regression',
        'description': 'Global longitudinal strain',
        'unit': '%',
        'normal_range': (-20, -16)
    },
    'LVEDV': {
        'type': 'regression',
        'description': 'LV end-diastolic volume',
        'unit': 'mL',
        'normal_range': (56, 104)
    },
    'LVESV': {
        'type': 'regression',
        'description': 'LV end-systolic volume',
        'unit': 'mL',
        'normal_range': (19, 49)
    },
    'LVSV': {
        'type': 'regression',
        'description': 'LV stroke volume',
        'unit': 'mL',
        'normal_range': (60, 100)
    },
    'LVSize': {
        'type': 'multi-class_classification',
        'classes': ['Normal', 'Mildly Increased', 'Moderately/Severely Increased'],
        'description': 'Left ventricular size',
        'unit': ''
    },
    'LVSystolicFunction': {
        'type': 'multi-class_classification',
        'classes': ['Normal/Hyperdynamic', 'Mildly Decreased', 'Moderately/Severely Decreased'],
        'description': 'LV systolic function',
        'unit': ''
    },
    'LVDiastolicFunction': {
        'type': 'multi-class_classification',
        'classes': ['Normal', 'Mild/Indeterminate', 'Moderate/Severe'],
        'description': 'LV diastolic function',
        'unit': ''
    },
    'RVSize': {
        'type': 'multi-class_classification',
        'classes': ['Normal', 'Mildly Increased', 'Moderately/Severely Increased'],
        'description': 'Right ventricular size',
        'unit': ''
    },
    'RVSystolicFunction': {
        'type': 'binary_classification',
        'classes': ['Normal', 'Decreased'],
        'description': 'RV systolic function',
        'unit': ''
    },
    'LASize': {
        'type': 'multi-class_classification',
        'classes': ['Normal', 'Mildly Dilated', 'Moderately/Severely Dilated'],
        'description': 'Left atrial size',
        'unit': ''
    },
    'RASize': {
        'type': 'binary_classification',
        'classes': ['Normal', 'Dilated'],
        'description': 'Right atrial size',
        'unit': ''
    },
    'AVStenosis': {
        'type': 'multi-class_classification',
        'classes': ['None', 'Mild/Moderate', 'Severe'],
        'description': 'Aortic valve stenosis',
        'unit': ''
    },
    'AVRegurg': {
        'type': 'multi-class_classification',
        'classes': ['None/Trace', 'Mild', 'Moderate/Severe'],
        'description': 'Aortic valve regurgitation',
        'unit': ''
    },
    'MVStenosis': {
        'type': 'binary_classification',
        'classes': ['Absent', 'Present'],
        'description': 'Mitral valve stenosis',
        'unit': ''
    },
    'MVRegurgitation': {
        'type': 'multi-class_classification',
        'classes': ['None/Trace', 'Mild', 'Moderate/Severe'],
        'description': 'Mitral valve regurgitation',
        'unit': ''
    },
    'TVRegurgitation': {
        'type': 'multi-class_classification',
        'classes': ['None/Trace', 'Mild', 'Moderate/Severe'],
        'description': 'Tricuspid valve regurgitation',
        'unit': ''
    },
    'TVPkGrad': {
        'type': 'regression',
        'description': 'Tricuspid valve peak gradient',
        'unit': 'mmHg',
        'normal_range': (0, 25)
    },
    'RAP': {
        'type': 'binary_classification',
        'classes': ['<8 mmHg', '≥8 mmHg'],
        'description': 'Right atrial pressure',
        'unit': ''
    },
    'AORoot': {
        'type': 'regression',
        'description': 'Aortic root diameter',
        'unit': 'cm',
        'normal_range': (2.0, 3.7)
    }
}

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load the PanEcho model"""
    return torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')

def run_model(model):
    """Run the model with random input and return results"""
    x = torch.rand(1, 3, 16, 224, 224)
    return model(x)

def safe_to_numpy(tensor):
    """Safely convert tensor to numpy, handling requires_grad"""
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def format_measurement(key: str, value: Any, metadata: Dict) -> Optional[Dict]:
    """Format a single measurement with metadata"""
    try:
        value_np = safe_to_numpy(value).flatten()
        
        result = {
            'parameter': key,
            'raw_value': value_np[0] if len(value_np) == 1 else value_np,
            'unit': metadata.get('unit', ''),
            'description': metadata.get('description', ''),
            'type': metadata.get('type', 'unknown'),
            'classes': metadata.get('classes', [])
        }
        
        if result['type'] == 'binary_classification':
            prob = float(value_np[0] if value_np.size > 0 else 0)
            class_idx = int(prob > 0.5)
            result.update({
                'formatted_value': f"{result['classes'][class_idx]} ({prob*100:.1f}%)",
                'is_abnormal': bool(class_idx),
                'value': prob,
                'class_name': result['classes'][class_idx] if result['classes'] else str(class_idx)
            })
        
        elif result['type'] == 'multi-class_classification':
            if value_np.size == 0:
                return None
            class_idx = int(np.argmax(value_np))
            prob = float(value_np[class_idx])
            classes = result['classes'] or [f'Class {i}' for i in range(len(value_np))]
            result.update({
                'formatted_value': f"{classes[class_idx]} ({prob*100:.1f}%)",
                'is_abnormal': class_idx != 0,  # Assuming first class is normal
                'value': prob,
                'class_name': classes[class_idx] if class_idx < len(classes) else f'Class {class_idx}',
                'probabilities': value_np.tolist()
            })
        
        else:  # regression
            value_float = float(value_np[0] if value_np.size > 0 else 0)
            normal_range = metadata.get('normal_range', (None, None))
            is_abnormal = (
                (normal_range[0] is not None and value_float < normal_range[0]) or
                (normal_range[1] is not None and value_float > normal_range[1])
            )
            result.update({
                'formatted_value': f"{value_float:.2f}{' ' + result['unit'] if result['unit'] else ''}",
                'is_abnormal': is_abnormal,
                'value': value_float,
                'normal_range': normal_range
            })
        
        return result
    except Exception as e:
        st.error(f"Error formatting {key}: {str(e)}")
        return None

def plot_gauge(value: float, title: str, min_val: float, max_val: float, normal_range: tuple):
    """Create a gauge chart for a single measurement"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, normal_range[0]], 'color': "lightgray"},
                {'range': normal_range, 'color': "lightgreen"},
                {'range': [normal_range[1], max_val], 'color': "lightgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(margin=dict(t=50, b=10, l=30, r=30))
    return fig

def plot_probability_bars(probabilities: list, classes: list, title: str):
    """Create a horizontal stacked bar chart for classification probabilities"""
    # Create a figure with a single trace that includes all probabilities
    fig = go.Figure()
    
    # Define colors based on class names (green for good, orange for intermediate, red for bad)
    color_map = {
        'normal': '#2ecc71',        # Green
        'none': '#2ecc71',          # Green
        'absent': '#2ecc71',        # Green
        'mild': '#f39c12',          # Orange
        'mildly': '#f39c12',        # Orange
        'moderate': '#e74c3c',      # Red
        'severe': '#c0392b',        # Darker red
        'dilated': '#e74c3c',       # Red
        'elevated': '#e74c3c',      # Red
        'present': '#e74c3c',       # Red
        'decreased': '#e74c3c'      # Red
    }
    
    # Add one trace for each class to create the stacked effect
    for i, (prob, cls) in enumerate(zip(probabilities, classes)):
        # Find the best matching color for the class
        color = '#3498db'  # Default blue if no match
        cls_lower = cls.lower()
        for key in color_map:
            if key in cls_lower:
                color = color_map[key]
                break
                
        fig.add_trace(go.Bar(
            y=[title],  # Single y-value for the stacked bar
            x=[prob * 100],  # Convert to percentage
            name=cls,   # Name for the legend
            text=f"{prob*100:.1f}% {cls}",  # Show value and class name
            textposition='inside',
            textfont=dict(color='white' if prob > 0.3 else 'black'),
            hovertemplate=f"{cls}: {prob*100:.1f}%<extra></extra>",
            marker_color=color,
            orientation='h'  # Make bars horizontal
        ))
    
    # Update layout for better appearance
    fig.update_layout(
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            title='Probability (%)',
            range=[0, 100],
            ticksuffix='%',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)'  # Lighter grid lines
        ),
        margin=dict(t=30, b=10, l=30, r=30),
        height=200,  # Reduced height since we're horizontal now
        hovermode='closest',
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent paper background
    )
    
    # Remove y-axis title and adjust tick labels
    fig.update_yaxes(showticklabels=False, title_text="")
    
    return fig

def create_ventricular_function_tab():
    """Create the Ventricular Function tab content"""
    if 'results' not in st.session_state:
        return
        
    results = st.session_state.results
    
    # --- Row 1: LV Dimensions Metrics ---
    st.subheader("LV Dimensions")
    lv_cols = st.columns(3)
    
    with lv_cols[0]:
        lvedv = next((r for r in results if r['parameter'] == 'LVEDV'), None)
        if lvedv:
            st.metric("LV End-Diastolic Volume", 
                     f"{lvedv['value']:.1f} {lvedv['unit']}",
                     delta=None if not lvedv['is_abnormal'] else "Abnormal",
                     delta_color="inverse")
    
    with lv_cols[1]:
        lvesv = next((r for r in results if r['parameter'] == 'LVESV'), None)
        if lvesv:
            st.metric("LV End-Systolic Volume",
                     f"{lvesv['value']:.1f} {lvesv['unit']}",
                     delta=None if not lvesv['is_abnormal'] else "Abnormal",
                     delta_color="inverse")
    
    with lv_cols[2]:
        lvsv = next((r for r in results if r['parameter'] == 'LVSV'), None)
        if lvsv:
            st.metric("Stroke Volume",
                     f"{lvsv['value']:.1f} {lvsv['unit']}",
                     delta=None if not lvsv['is_abnormal'] else "Abnormal",
                     delta_color="inverse")
    
    # --- Row 2: Gauge Charts ---
    st.markdown("---")
    st.subheader("Left Ventricular Function")
    gauge_cols = st.columns(2)
    
    with gauge_cols[0]:
        ef = next((r for r in results if r['parameter'] == 'EF'), None)
        if ef:
            st.plotly_chart(
                plot_gauge(
                    ef['value'], 
                    'Ejection Fraction', 
                    0, 100, 
                    ef.get('normal_range', (55, 70))
                ), 
                use_container_width=True
            )
    
    with gauge_cols[1]:
        gls = next((r for r in results if r['parameter'] == 'GLS'), None)
        if gls:
            st.plotly_chart(
                plot_gauge(
                    gls['value'],
                    'Global Longitudinal Strain',
                    -30, 0,
                    gls.get('normal_range', (-20, -16))
                ),
                use_container_width=True
            )
    
    # --- Row 3: LV Size Classification ---
    st.markdown("---")
    st.subheader("LV Size Classification")
    lv_size = next((r for r in results if r['parameter'] == 'LVSize'), None)
    if lv_size and 'probabilities' in lv_size:
        st.plotly_chart(
            plot_probability_bars(
                lv_size['probabilities'],
                lv_size['classes'],
                'LV Size Classification'
            ),
            use_container_width=True
        )

def create_valvular_assessment_tab():
    """Create the Valvular Assessment tab content"""
    if 'results' not in st.session_state:
        return
        
    results = st.session_state.results
    
    # --- Numerical Metrics at the Top ---
    st.subheader("Valvular Metrics")
    
    # Create a single row for all valve metrics
    metric_cols = st.columns(3)
    
    # Aortic Valve Metric
    with metric_cols[0]:
        st.markdown("#### Aortic Valve")
        av_vel = next((r for r in results if r['parameter'] == 'AVPkVel(m|s)'), None)
        if av_vel:
            st.metric("Peak Velocity",
                     f"{av_vel['value']:.2f} {av_vel['unit']}",
                     delta=None if not av_vel['is_abnormal'] else "Abnormal",
                     delta_color="inverse")
    
    # Mitral Valve Metric
    with metric_cols[1]:
        st.markdown("#### Mitral Valve")
        mv_stenosis = next((r for r in results if r['parameter'] == 'MVStenosis'), None)
        if mv_stenosis:
            st.metric("Stenosis",
                     mv_stenosis['formatted_value'],
                     delta=None if not mv_stenosis['is_abnormal'] else "Present",
                     delta_color="inverse")
    
    # Tricuspid Valve Metric
    with metric_cols[2]:
        st.markdown("#### Tricuspid Valve")
        tv_grad = next((r for r in results if r['parameter'] == 'TVPkGrad'), None)
        if tv_grad:
            st.metric("Peak Gradient",
                     f"{tv_grad['value']:.1f} {tv_grad['unit']}",
                     delta=None if not tv_grad['is_abnormal'] else "Elevated",
                     delta_color="inverse")
    
    # Add some vertical space before the plots
    st.markdown("")
    
    # --- Stacked Bar Plots Below ---
    st.markdown("---")
    st.subheader("Valvular Function")
    
    # Create a 2x2 grid for the plots
    plot_cols = st.columns(2)
    
    with plot_cols[0]:
        # Aortic Valve Plots
        st.markdown("##### Aortic Stenosis")
        
        # Aortic Stenosis
        av_stenosis = next((r for r in results if r['parameter'] == 'AVStenosis'), None)
        if av_stenosis and 'probabilities' in av_stenosis:
            st.plotly_chart(
                plot_probability_bars(
                    av_stenosis['probabilities'],
                    av_stenosis['classes'],
                    'Aortic Valve Stenosis'
                ),
                use_container_width=True
            )
        
        st.markdown("##### Aortic Regurgitation")
        av_regurg = next((r for r in results if r['parameter'] == 'AVRegurg'), None)
        if av_regurg and 'probabilities' in av_regurg:
            st.plotly_chart(
                plot_probability_bars(
                    av_regurg['probabilities'],
                    av_regurg['classes'],
                    'Aortic Valve Regurgitation'
                ),
                use_container_width=True
            )
    
    with plot_cols[1]:
        # Mitral Valve Plots
        st.markdown("##### Mitral Regurgitation")
        
        # Mitral Regurgitation
        mv_regurg = next((r for r in results if r['parameter'] == 'MVRegurgitation'), None)
        if mv_regurg and 'probabilities' in mv_regurg:
            st.plotly_chart(
                plot_probability_bars(
                    mv_regurg['probabilities'],
                    mv_regurg['classes'],
                    'Mitral Valve Regurgitation'
                ),
                use_container_width=True
            )
        
        # Tricuspid Valve Plots
        st.markdown("##### Tricuspid Regurgitation")
        
        # Tricuspid Regurgitation
        tv_regurg = next((r for r in results if r['parameter'] == 'TVRegurgitation'), None)
        if tv_regurg and 'probabilities' in tv_regurg:
            st.plotly_chart(
                plot_probability_bars(
                    tv_regurg['probabilities'],
                    tv_regurg['classes'],
                    'Tricuspid Valve Regurgitation'
                ),
                use_container_width=True
            )

def create_atrial_function_tab():
    """Create the Atrial Function tab content"""
    if 'results' not in st.session_state:
        return
        
    results = st.session_state.results
    
    # Create a single row for all atrial metrics
    st.subheader("Atrial Function")
    
    # Left Atrium
    la_cols = st.columns(2)
    
    with la_cols[0]:
        st.markdown("#### Left Atrium")
        la_size = next((r for r in results if r['parameter'] == 'LASize'), None)
        if la_size and 'probabilities' in la_size:
            st.plotly_chart(
                plot_probability_bars(
                    la_size['probabilities'],
                    la_size['classes'],
                    'Left Atrial Size'
                ),
                use_container_width=True
            )
    
    # Right Atrium
    with la_cols[1]:
        st.markdown("#### Right Atrium")
        ra_cols = st.columns(2)
        
        with ra_cols[0]:
            ra_size = next((r for r in results if r['parameter'] == 'RASize'), None)
            if ra_size:
                st.metric("Right Atrial Size",
                         ra_size['formatted_value'],
                         delta=None if not ra_size['is_abnormal'] else "Dilated",
                         delta_color="inverse")
        
        with ra_cols[1]:
            rap = next((r for r in results if r['parameter'] == 'RAP-8-or-higher'), None)
            if rap:
                st.metric("Right Atrial Pressure",
                         rap['formatted_value'],
                         delta=None if not rap['is_abnormal'] else "Elevated",
                         delta_color="inverse")

def create_full_report_tab():
    """Create the Full Report tab content"""
    if 'results' not in st.session_state:
        return
        
    results_df = pd.DataFrame(st.session_state.results)
    
    # Display the full data table
    st.dataframe(
        results_df[['parameter', 'formatted_value', 'description', 'is_abnormal']],
        column_config={
            'parameter': 'Parameter',
            'formatted_value': 'Value',
            'description': 'Description',
            'is_abnormal': st.column_config.CheckboxColumn('Abnormal')
        },
        hide_index=True,
        width='stretch'
    )
    
    # Download button for full report
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name=f'echo_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

def main():
    st.set_page_config(
        page_title="Enigma - Echocardiogram Analysis",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "<h1 style='text-align:center;'>Engima AI</h1>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        if st.button("Run Differentiatial Diagnosis", width='stretch', type='primary', use_container_width=True):
            st.session_state.run_analysis = True
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Interactive dashboard for echocardiogram analysis.
        
        **Features:**
        - Real-time visualization of echocardiac measurements
        - Interactive charts and gauges
        - Detailed reports
        """)

    # --- Main Content ---
    st.title("Echocardiogram Analysis Dashboard")
    
    # Load model if not already loaded
    if 'model' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model = load_model()
    
    # Run analysis if button was clicked
    if st.session_state.get('run_analysis', False):
        with st.spinner("Analyzing echocardiogram data..."):
            start_time = time.time()
            results = run_model(st.session_state.model)
            
            # Process and store results
            processed_results = []
            for key, value in results.items():
                metadata = MODEL_METADATA.get(key, {'type': 'regression'})
                formatted = format_measurement(key, value, metadata)
                if formatted:
                    processed_results.append(formatted)
            
            st.session_state.results = processed_results
            st.session_state.analysis_time = time.time() - start_time
            st.session_state.last_updated = time.time()
            st.session_state.run_analysis = False
            st.rerun()
    
    # Display results if available
    if 'results' in st.session_state and st.session_state.results:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Ventricular Function", "Valvular Assessment", "Atrial Function", "Full Report"])
        
        with tab1:
            create_ventricular_function_tab()
            
        with tab2:
            create_valvular_assessment_tab()
            
        with tab3:
            create_atrial_function_tab()
            
        with tab4:
            create_full_report_tab()
    
    else:
        # Initial state - show welcome/instructions
        st.info("👈 Click 'Run Analysis' in the sidebar to start the echocardiogram analysis.")
        
        # Add some space and a brief description
        st.markdown("---")
        st.markdown("### Getting Started")
        st.markdown("""
        1. Click the 'Run Analysis' button in the sidebar to process a sample echocardiogram
        2. Explore the different tabs to view detailed metrics
        3. Download a full report from the 'Full Report' tab
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    # Run the app
    main()

