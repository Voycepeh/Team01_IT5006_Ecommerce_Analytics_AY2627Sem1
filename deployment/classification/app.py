"""Standalone post-delivery classification POC. Run from the repository root."""
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.classification_app import load_model, category_options, predict_order
from src.classification_data import DELIVERED_NUMERIC_FEATURES

st.set_page_config(page_title='Order review risk | IT5006', page_icon='📦')
st.title('Which delivered orders need a follow-up?')
st.write('Estimate the risk of a 1–2 star review to help customer service prioritise outreach.')
st.info('Use only after delivery and before a review has been submitted. This is a historical Olist proof of concept, not a checkout-time prediction.')

@st.cache_resource
def model():
    return load_model()

try:
    artifact = model()
    options = category_options(artifact)
except Exception:
    logging.exception('Classification artifact could not be loaded')
    st.error('The model could not be loaded. Ask the maintainer to check model.joblib and the pinned dependencies. No prediction was made.')
    st.stop()

labels = {
    'item_count': 'Number of items', 'order_item_value': 'Total item value (BRL)',
    'order_freight_value': 'Total freight (BRL)', 'seller_count': 'Number of sellers',
    'payment_total': 'Total paid (BRL)', 'payment_count': 'Number of payment records',
    'max_installments': 'Maximum instalments', 'delivery_days': 'Purchase to delivery (whole days)',
    'days_early': 'Days early (negative means late)', 'n_seller_states': 'Number of seller states',
    'product_weight_g_mean': 'Mean item weight (g)',
    'product_volume_cm3_mean': 'Mean item volume (cm³)',
    'product_photos_qty_mean': 'Mean number of product photos',
}
defaults = {'item_count': 1, 'seller_count': 1, 'payment_count': 1, 'max_installments': 1,
            'n_seller_states': 1, 'order_item_value': 100., 'order_freight_value': 20.,
            'payment_total': 120., 'delivery_days': 10., 'days_early': 5.,
            'product_weight_g_mean': 500., 'product_volume_cm3_mean': 1000.,
            'product_photos_qty_mean': 1.}
with st.form('order'):
    eligible = st.checkbox('This order is delivered and has not been reviewed')
    st.caption('Enter order totals and item-level means. For multiple sellers, use the first seller state from the order table. Category means the category with the greatest total item value; break ties alphabetically.')
    values = {}
    for field in DELIVERED_NUMERIC_FEATURES:
        if field in {'freight_share', 'late_delivery_flag'}:
            continue
        default = defaults[field]
        values[field] = st.number_input(labels[field], value=default,
                                       min_value=None if field == 'days_early' else type(default)(0),
                                       step=1 if isinstance(default, int) else 1.)
    for field in ['customer_state', 'seller_state', 'primary_product_category', 'purchase_month', 'purchase_weekday']:
        values[field] = st.selectbox(('First seller state' if field == 'seller_state' else field.replace('_', ' ').capitalize()), options[field])
    st.caption('Purchase month includes the year (YYYY-MM), matching the historical training data. Weekday refers to purchase day. Future months require model review rather than silently treating them as known.')
    submitted = st.form_submit_button('Assess review risk', type='primary')
if submitted:
    if not eligible:
        st.error('Confirm that the order is delivered and has not been reviewed.')
    else:
        total = values['order_item_value'] + values['order_freight_value']
        values['freight_share'] = values['order_freight_value'] / total if total else 0.
        values['late_delivery_flag'] = int(values['days_early'] < 0)
        values['same_state'] = values['customer_state'] == values['seller_state']
        try:
            result = predict_order(artifact, values)
            st.metric('Estimated negative-review risk', f"{result['negative_review_probability']:.1%}")
            if result['predicted_negative_review']:
                st.warning('Prioritise for follow-up: risk meets the 12% contact cutoff.')
            else:
                st.success('Below the 12% contact cutoff. A negative review is still possible.')
            st.write('Customer service can check delivery experience and offer help. A flag does not prove dissatisfaction or that outreach will prevent a negative review.')
        except ValueError as error:
            st.error(str(error))
        except Exception:
            logging.exception('Classification inference failed')
            st.error('Prediction failed. Please ask the maintainer to check the model and inputs.')
with st.expander('Model, policy and limitations'):
    st.write('Tuned Logistic Regression: C=0.01, class_weight=None; preprocessing and cutoff are frozen with the fitted model. The illustrative cutoff was selected on validation data for at least 20% precision, not approved by a business stakeholder.')
    st.write('On 18,235 later historical orders: 24.8% precision and 22.2% recall; 1,385 flagged, 343 negative reviews detected, 1,042 false alarms and 1,200 missed. Roughly three quarters of flags were false alarms. These are historical results, not guarantees for a new order.')
    st.write('The score is a model estimate; calibration has not been established. Data covers 2016–2018, excludes pre-delivery reviews, and uses purchase-time splitting as a proxy. Monitor drift before operational use.')
