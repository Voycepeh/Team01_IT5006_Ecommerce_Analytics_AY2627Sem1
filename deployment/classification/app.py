"""Standalone post-delivery classification POC. Run from the repository root."""
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.classification_app import load_model, category_options, predict_order

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

def number(field, label, default, *, help=None, minimum=0):
    """Give related controls consistent labels and stable widget keys."""
    return st.number_input(label, value=default, min_value=None if minimum is None else type(default)(minimum),
                           step=1 if isinstance(default, int) else 1.,
                           key=field, help=help)


def category(field, label, *, help=None):
    choices = options[field]
    # Start with the fitted training mode rather than the alphabetical first value.
    preprocessor = artifact['pipeline'].named_steps['preprocessor']
    names = preprocessor.transformers_[1][2]
    modes = preprocessor.named_transformers_['categorical'].named_steps['imputer'].statistics_
    default = modes[list(names).index(field)]
    return st.selectbox(label, choices, index=choices.index(default), key=field, help=help)


values = {}
with st.form('order', border=False):
    st.subheader('Order details')
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            values['item_count'] = number('item_count', 'Number of items', 1)
            values['order_item_value'] = number('order_item_value', 'Item total (BRL)', 100.,
                                               help='Sum of item prices, excluding freight.')
        with right:
            values['seller_count'] = number('seller_count', 'Number of sellers', 1,
                                            help='Distinct sellers on this order; cannot exceed the item count.')
            values['order_freight_value'] = number('order_freight_value', 'Freight total (BRL)', 20.)
        values['primary_product_category'] = category(
            'primary_product_category', 'Main product category',
            help='Category with the largest sum of item prices. Break equal-value ties alphabetically.')

    st.subheader('Delivery and location')
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            values['delivery_days'] = number('delivery_days', 'Purchase to delivery (days)', 10,
                                             help='Whole elapsed days from purchase until customer delivery.')
            values['customer_state'] = category('customer_state', 'Customer state')
        with right:
            values['days_early'] = number('days_early', 'Days ahead of promise', 5, minimum=None,
                                          help='5 means five days early; 0 means on the promised day; −5 means five days late.')
            values['seller_state'] = category('seller_state', 'First seller state',
                                               help='For multiple sellers, use the first seller state in the order-level data.')
        st.caption('Lateness, same-state status and freight share are calculated automatically.')

    with st.expander('Additional details: payment, products and purchase context', expanded=False):
        st.caption('These inputs still affect the prediction. The example defaults below are used unless you change them; they are not facts about your order.')
        st.markdown('**Payment**')
        left, right = st.columns(2)
        with left:
            paid = st.number_input('Total paid (BRL)', value=None, min_value=0.,
                                   placeholder='Use item total + freight', key='payment_total',
                                   help='Leave blank to use item total plus freight. Override for a different recorded payment total.')
            values['payment_count'] = number('payment_count', 'Number of payment records', 1,
                                              help='Default: one payment record, not the number of instalments.')
        with right:
            values['max_installments'] = number('max_installments', 'Maximum instalments', 1,
                                                 help='Largest instalment count across payment records. Zero is valid if recorded.')
            values['n_seller_states'] = number('n_seller_states', 'Distinct seller states', 1,
                                                help='Default: one seller state. Increase if sellers span multiple states.')
        st.markdown('**Product averages per item**')
        left, right = st.columns(2)
        with left:
            values['product_weight_g_mean'] = number('product_weight_g_mean', 'Mean item weight (g)', 500.)
            values['product_photos_qty_mean'] = number('product_photos_qty_mean', 'Mean product photos', 1.)
        with right:
            values['product_volume_cm3_mean'] = number('product_volume_cm3_mean', 'Mean item volume (cm³)', 1000.,
                                                       help='Calculate length × height × width for each item, then average those volumes.')
        st.markdown('**Historical purchase context**')
        left, right = st.columns(2)
        with left:
            values['purchase_month'] = category('purchase_month', 'Purchase month')
        with right:
            values['purchase_weekday'] = category('purchase_weekday', 'Purchase weekday')
        st.caption('Month includes the year. Choices are restricted to the historical training categories; this POC is not intended for current live orders.')

    st.caption('Additional defaults: 1 payment record, 1 instalment, 1 seller state; product averages of 500 g, 1,000 cm³ and 1 photo. Purchase context starts at the most common training values. Expand Additional details to edit them.')
    eligible = st.checkbox('I confirm that this order is completed and no review has been submitted.', key='eligible')
    st.caption('This is your confirmation of the intended use. The model does not receive review information and cannot verify this status.')
    submitted = st.form_submit_button('Assess review risk', type='primary')

if submitted:
    if not eligible:
        st.error('Confirm that the order is delivered and has not been reviewed.')
    else:
        total = values['order_item_value'] + values['order_freight_value']
        values['payment_total'] = total if paid is None else paid
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
