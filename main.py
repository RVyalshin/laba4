#!/usr/bin/env python3
"""
Финансовый калькулятор для Telegram
Полная реализация на python-telegram-bot
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, asdict
import math
import random
from io import BytesIO
import html

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from dotenv import load_dotenv
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    InputFile
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, ConversationHandler, filters
)
from telegram.constants import ParseMode
from scipy import stats

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"

user_settings = {}
calculation_history = {}

AMOUNT, YEARS, RATE, INVESTMENT, CAPITALIZATION, LOAN_TYPE = range(6)

class CalculationType(Enum):
    LOAN = "loan"
    DEPOSIT = "deposit"
    INVESTMENT = "investment"
    CURRENCY = "currency"


@dataclass
class UserSettings:
    user_id: int
    default_currency: str = "RUB"
    notifications: bool = True
    language: str = "ru"
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class CalculationRecord:
    calc_type: CalculationType
    params: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FinancialCalculator:

    @staticmethod
    def calculate_loan(
            amount: float,
            years: int,
            annual_rate: float,
            loan_type: str = "annuity"
    ) -> Dict[str, Any]:
        try:
            if amount <= 0 or years <= 0 or annual_rate <= 0:
                raise ValueError("Все значения должны быть положительными")

            months = years * 12
            monthly_rate = annual_rate / 100 / 12

            if loan_type == "annuity":
                if monthly_rate == 0:
                    monthly_payment = amount / months
                else:
                    coeff = (monthly_rate * (1 + monthly_rate) ** months) / \
                            ((1 + monthly_rate) ** months - 1)
                    monthly_payment = amount * coeff

                total_payment = monthly_payment * months
                overpayment = total_payment - amount

                schedule = []
                remaining = amount
                for month in range(1, min(7, months + 1)):
                    interest = remaining * monthly_rate
                    principal = monthly_payment - interest
                    remaining -= principal
                    schedule.append({
                        "month": month,
                        "payment": round(monthly_payment, 2),
                        "principal": round(principal, 2),
                        "interest": round(interest, 2),
                        "remaining": round(max(remaining, 0), 2)
                    })

                return {
                    "success": True,
                    "monthly_payment": round(monthly_payment, 2),
                    "total_payment": round(total_payment, 2),
                    "overpayment": round(overpayment, 2),
                    "overpayment_percent": round((overpayment / amount) * 100, 2),
                    "schedule": schedule,
                    "loan_type": loan_type,
                    "months": months
                }

            elif loan_type == "differentiated":
                principal_payment = amount / months
                schedule = []
                total_payment = 0

                remaining = amount
                for month in range(1, min(7, months + 1)):
                    interest = remaining * monthly_rate
                    monthly_payment = principal_payment + interest
                    remaining -= principal_payment
                    total_payment += monthly_payment

                    schedule.append({
                        "month": month,
                        "payment": round(monthly_payment, 2),
                        "principal": round(principal_payment, 2),
                        "interest": round(interest, 2),
                        "remaining": round(max(remaining, 0), 2)
                    })

                overpayment = total_payment * (months / min(7, months)) - amount
                total_payment = amount + overpayment

                return {
                    "success": True,
                    "first_payment": round(schedule[0]["payment"], 2),
                    "last_payment": round(
                        principal_payment + (amount - principal_payment * (months - 1)) * monthly_rate, 2),
                    "total_payment": round(total_payment, 2),
                    "overpayment": round(overpayment, 2),
                    "overpayment_percent": round((overpayment / amount) * 100, 2),
                    "schedule": schedule,
                    "loan_type": loan_type,
                    "months": months
                }

            else:
                raise ValueError("Неизвестный тип платежей")

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def calculate_deposit(
            amount: float,
            years: int,
            annual_rate: float,
            capitalization: str = "monthly"
    ) -> Dict[str, Any]:
        try:
            if amount <= 0 or years <= 0 or annual_rate <= 0:
                raise ValueError("Все значения должны быть положительными")

            if capitalization == "monthly":
                periods_per_year = 12
            elif capitalization == "quarterly":
                periods_per_year = 4
            elif capitalization == "yearly":
                periods_per_year = 1
            elif capitalization == "end":
                periods_per_year = 1
            else:
                raise ValueError("Неизвестный тип капитализации")

            total_periods = years * periods_per_year
            period_rate = annual_rate / 100 / periods_per_year

            if capitalization == "end":
                interest = amount * annual_rate / 100 * years
                final_amount = amount + interest
            else:
                final_amount = amount * (1 + period_rate) ** total_periods
                interest = final_amount - amount

            tax_free_rate = 0.0425
            tax_rate = 0.13
            tax_base = max(interest - tax_free_rate * years, 0)
            tax = tax_base * tax_rate

            return {
                "success": True,
                "final_amount": round(final_amount, 2),
                "interest": round(interest, 2),
                "tax": round(tax, 2),
                "capitalization": capitalization,
                "years": years
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def calculate_investment(
            initial_amount: float,
            monthly_investment: float,
            years: int,
            annual_return: float
    ) -> Dict[str, Any]:
        """
        Расчет инвестиций со сложным процентом
        """
        try:
            if annual_return < -100:
                raise ValueError("Доходность не может быть меньше -100%")

            months = years * 12
            monthly_return = annual_return / 100 / 12
            total_invested = initial_amount

            amounts = []
            current_amount = initial_amount

            for month in range(1, months + 1):
                if month > 1:
                    current_amount += monthly_investment
                    total_invested += monthly_investment

                current_amount *= (1 + monthly_return)

                if month % 12 == 0 or month == months:
                    amounts.append({
                        "year": month // 12,
                        "amount": round(current_amount, 2),
                        "invested": round(total_invested, 2),
                        "profit": round(current_amount - total_invested, 2)
                    })

            final_amount = current_amount
            total_profit = final_amount - total_invested

            return {
                "success": True,
                "final_amount": round(final_amount, 2),
                "total_invested": round(total_invested, 2),
                "total_profit": round(total_profit, 2),
                "profit_percent": round((total_profit / total_invested) * 100, 2) if total_invested > 0 else 0,
                "yearly_results": amounts,
                "years": years
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class DataAnalyzer:

    @staticmethod
    def generate_sample_data() -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        n = len(dates)

        data = {
            'date': dates,
            'revenue': np.random.normal(100000, 20000, n).cumsum(),
            'expenses': np.random.normal(60000, 15000, n).cumsum(),
            'profit': np.zeros(n),
            'investments': np.random.exponential(5000, n).cumsum(),
            'interest_rate': np.random.uniform(3, 12, n)
        }

        df = pd.DataFrame(data)
        df['profit'] = df['revenue'] - df['expenses']
        df['profit_margin'] = (df['profit'] / df['revenue']) * 100
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        return df

    @staticmethod
    def create_visualizations(df: pd.DataFrame) -> List[BytesIO]:
        images = []

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df['date'], df['profit'], 'b-', linewidth=2)
            plt.title('Динамика прибыли по месяцам', fontsize=14)
            plt.xlabel('Дата', fontsize=12)
            plt.ylabel('Прибыль', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            images.append(img_buf)
            plt.close()
            plt.figure(figsize=(10, 6))
            plt.hist(df['interest_rate'], bins=15, edgecolor='black', alpha=0.7)
            plt.title('Распределение процентных ставок', fontsize=14)
            plt.xlabel('Процентная ставка (%)', fontsize=12)
            plt.ylabel('Частота', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            images.append(img_buf)
            plt.close()

            plt.figure(figsize=(10, 6))
            numeric_cols = ['revenue', 'expenses', 'profit', 'investments', 'interest_rate']
            corr_matrix = df[numeric_cols].corr()

            plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Корреляция')
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title('Корреляционная матрица финансовых показателей', fontsize=14)

            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')

            plt.tight_layout()
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            images.append(img_buf)
            plt.close()

            plt.figure(figsize=(10, 6))
            df.boxplot(column='profit', by='year', grid=True)
            plt.title('Распределение прибыли по годам', fontsize=14)
            plt.suptitle('')
            plt.xlabel('Год', fontsize=12)
            plt.ylabel('Прибыль', fontsize=12)
            plt.tight_layout()

            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            images.append(img_buf)
            plt.close()

        except Exception as e:
            logging.error(f"Ошибка при создании визуализаций: {e}")

        return images

    @staticmethod
    def test_statistical_hypothesis(df: pd.DataFrame) -> Dict[str, Any]:

        try:
            profit_data = df['profit'].dropna()

            if len(profit_data) < 3:
                raise ValueError("Недостаточно данных для анализа")

            if len(profit_data) <= 5000:
                stat, p_value = stats.shapiro(profit_data)
                test_name = "Shapiro-Wilk"
            else:
                stat, p_value = stats.kstest(profit_data, 'norm',
                                             args=(profit_data.mean(), profit_data.std()))
                test_name = "Kolmogorov-Smirnov"

            skewness = stats.skew(profit_data)
            kurtosis = stats.kurtosis(profit_data)

            alpha = 0.05
            is_normal = p_value > alpha
            interpretation = "нормальное" if is_normal else "не нормальное"

            plt.figure(figsize=(8, 6))
            stats.probplot(profit_data, dist="norm", plot=plt)
            plt.title('Q-Q Plot для проверки нормальности распределения прибыли')
            plt.tight_layout()

            qq_buf = BytesIO()
            plt.savefig(qq_buf, format='png', dpi=100)
            qq_buf.seek(0)
            plt.close()

            return {
                "success": True,
                "test_name": test_name,
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "is_normal": is_normal,
                "interpretation": interpretation,
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "mean": round(profit_data.mean(), 2),
                "std": round(profit_data.std(), 2),
                "sample_size": len(profit_data),
                "qq_plot": qq_buf
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class Keyboards:

    @staticmethod
    def get_main_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("💰 Кредитный калькулятор", callback_data="calc_loan")],
            [InlineKeyboardButton("💳 Калькулятор вклада", callback_data="calc_deposit")],
            [InlineKeyboardButton("📈 Инвестиционный калькулятор", callback_data="calc_investment")],
            [InlineKeyboardButton("📊 Анализ финансовых данных", callback_data="analysis")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")],
            [InlineKeyboardButton("📋 История расчетов", callback_data="history")],
            [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def get_calc_types(calc_type: str) -> InlineKeyboardMarkup:
        keyboard = []

        if calc_type == "loan":
            keyboard = [
                [InlineKeyboardButton("Аннуитетные платежи", callback_data="loan_annuity")],
                [InlineKeyboardButton("Дифференцированные платежи", callback_data="loan_differentiated")],
                [InlineKeyboardButton("◀️ Назад", callback_data="back_to_menu")]
            ]
        elif calc_type == "deposit":
            keyboard = [
                [InlineKeyboardButton("С ежемесячной капитализацией", callback_data="deposit_monthly")],
                [InlineKeyboardButton("С ежеквартальной капитализацией", callback_data="deposit_quarterly")],
                [InlineKeyboardButton("С ежегодной капитализацией", callback_data="deposit_yearly")],
                [InlineKeyboardButton("Без капитализации", callback_data="deposit_end")],
                [InlineKeyboardButton("◀️ Назад", callback_data="back_to_menu")]
            ]
        elif calc_type == "investment":
            keyboard = [
                [InlineKeyboardButton("Рассчитать", callback_data="investment_calc")],
                [InlineKeyboardButton("◀️ Назад", callback_data="back_to_menu")]
            ]

        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def get_settings_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🌍 Валюта (RUB)", callback_data="set_currency")],
            [InlineKeyboardButton("🔔 Уведомления (Вкл)", callback_data="toggle_notifications")],
            [InlineKeyboardButton("◀️ Назад", callback_data="back_to_menu")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def get_currency_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("RUB", callback_data="currency_RUB"),
                InlineKeyboardButton("USD", callback_data="currency_USD"),
                InlineKeyboardButton("EUR", callback_data="currency_EUR")
            ],
            [
                InlineKeyboardButton("KZT", callback_data="currency_KZT"),
                InlineKeyboardButton("BYN", callback_data="currency_BYN")
            ],
            [InlineKeyboardButton("◀️ Назад", callback_data="settings")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def get_back_button() -> InlineKeyboardMarkup:
        keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="back_to_menu")]]
        return InlineKeyboardMarkup(keyboard)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_settings:
        user_settings[user_id] = UserSettings(user_id=user_id)
        calculation_history[user_id] = []

    welcome_text = f"""
    👋 <b>Добро пожаловать в Финансовый калькулятор!</b>

    Я помогу вам с расчетами:
    • 💰 <b>Кредитов</b> (аннуитетные/дифференцированные платежи)
    • 💳 <b>Вкладов</b> с разной капитализацией
    • 📈 <b>Инвестиций</b> со сложным процентом
    • 📊 <b>Анализа финансовых данных</b>

    Выберите действие из меню ниже:
    """

    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.HTML,
        reply_markup=Keyboards.get_main_menu()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
    <b>📚 Помощь по использованию бота</b>

    <b>Основные команды:</b>
    /start - Запустить бота
    /help - Показать эту справку
    /loan - Рассчитать кредит
    /deposit - Рассчитать вклад
    /investment - Рассчитать инвестиции
    /analysis - Анализ финансовых данных
    /settings - Настройки

    <b>Как использовать:</b>
    1. Выберите тип расчета из меню
    2. Следуйте инструкциям бота
    3. Вводите данные по запросу

    <b>Формат ввода данных:</b>
    • Числа можно вводить с точкой или запятой: 100000 или 100,000
    • Проценты вводятся как число: 15 (для 15%)
    • Срок в годах: 5 (для 5 лет)

    <b>Примеры быстрого ввода:</b>
    Кредит: 1000000 5 15
    Вклад: 500000 3 7
    Инвестиции: 100000 5000 10 12
    """

    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.HTML,
        reply_markup=Keyboards.get_back_button()
    )


# ========== ОБРАБОТЧИКИ КНОПОК ==========
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "calc_loan":
        await query.edit_message_text(
            "💰 <b>Кредитный калькулятор</b>\n\nВыберите тип платежей:",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_calc_types("loan")
        )
    elif data.startswith("loan_"):
        loan_type = "annuity" if "annuity" in data else "differentiated"
        context.user_data['loan_type'] = loan_type
        context.user_data['calc_type'] = 'loan'

        await query.edit_message_text(
            f"💰 <b>Расчет {loan_type} кредита</b>\n\n"
            "Введите сумму кредита (например: 1000000):",
            parse_mode=ParseMode.HTML
        )
        return AMOUNT
    elif data == "calc_deposit":
        await query.edit_message_text(
            "💳 <b>Калькулятор вклада</b>\n\nВыберите тип капитализации:",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_calc_types("deposit")
        )
    elif data.startswith("deposit_"):
        capitalization = data.replace("deposit_", "")
        context.user_data['capitalization'] = capitalization
        context.user_data['calc_type'] = 'deposit'

        await query.edit_message_text(
            f"💳 <b>Расчет вклада с {capitalization} капитализацией</b>\n\n"
            "Введите сумму вклада (например: 100000):",
            parse_mode=ParseMode.HTML
        )
        return AMOUNT
    elif data == "calc_investment":
        await query.edit_message_text(
            "📈 <b>Инвестиционный калькулятор</b>\n\n"
            "Нажмите Рассчитать для начала:",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_calc_types("investment")
        )
    elif data == "investment_calc":
        context.user_data['calc_type'] = 'investment'
        await query.edit_message_text(
            "📈 <b>Расчет инвестиций</b>\n\n"
            "Введите начальную сумму инвестиций (например: 100000):",
            parse_mode=ParseMode.HTML
        )
        return AMOUNT
    elif data == "analysis":
        await query.edit_message_text(
            "📊 <b>Анализ финансовых данных</b>\n\n"
            "Генерирую тестовые данные и выполняю анализ...",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_back_button()
        )
        await perform_data_analysis(query.message, context)
    elif data == "settings":
        user_id = query.from_user.id
        settings = user_settings.get(user_id, UserSettings(user_id=user_id))

        settings_text = f"""
        ⚙️ <b>Настройки</b>

        Текущие настройки:
        • Валюта: {settings.default_currency}
        • Уведомления: {'Включены' if settings.notifications else 'Выключены'}
        • Язык: {settings.language}
        • Дата регистрации: {settings.created_at[:10]}
        """

        await query.edit_message_text(
            settings_text,
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_settings_menu()
        )
    elif data == "set_currency":
        await query.edit_message_text(
            "🌍 <b>Выберите валюту:</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_currency_menu()
        )
    elif data.startswith("currency_"):
        currency = data.replace("currency_", "")
        user_id = query.from_user.id

        if user_id in user_settings:
            user_settings[user_id].default_currency = currency

        await query.edit_message_text(
            f"✅ Валюта изменена на {currency}",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_settings_menu()
        )
    elif data == "toggle_notifications":
        user_id = query.from_user.id

        if user_id in user_settings:
            user_settings[user_id].notifications = not user_settings[user_id].notifications
            status = "включены" if user_settings[user_id].notifications else "выключены"

            await query.edit_message_text(
                f"✅ Уведомления {status}",
                parse_mode=ParseMode.HTML,
                reply_markup=Keyboards.get_settings_menu()
            )
    elif data == "history":
        user_id = query.from_user.id

        if user_id not in calculation_history or not calculation_history[user_id]:
            await query.edit_message_text(
                "📋 <b>История расчетов пуста</b>\n\n"
                "Выполните хотя бы один расчет, чтобы увидеть историю.",
                parse_mode=ParseMode.HTML,
                reply_markup=Keyboards.get_back_button()
            )
            return

        history = calculation_history[user_id][-10:]
        history_text = "📋 <b>История расчетов (последние 10):</b>\n\n"

        for i, record in enumerate(reversed(history), 1):
            date_str = datetime.fromisoformat(record.timestamp).strftime("%d.%m.%Y %H:%M")

            if record.calc_type == CalculationType.LOAN:
                params = record.params
                history_text += f"{i}. 💰 <b>Кредит</b> ({date_str})\n"
                history_text += f"   Сумма: {params['amount']:,.0f}, "
                history_text += f"Срок: {params['years']} лет, "
                history_text += f"Ставка: {params['rate']}%\n\n"
            elif record.calc_type == CalculationType.DEPOSIT:
                params = record.params
                history_text += f"{i}. 💳 <b>Вклад</b> ({date_str})\n"
                history_text += f"   Сумма: {params['amount']:,.0f}, "
                history_text += f"Срок: {params['years']} лет, "
                history_text += f"Ставка: {params['rate']}%\n\n"
            elif record.calc_type == CalculationType.INVESTMENT:
                params = record.params
                history_text += f"{i}. 📈 <b>Инвестиции</b> ({date_str})\n"
                history_text += f"   Начальная сумма: {params['initial']:,.0f}, "
                history_text += f"Срок: {params['years']} лет\n\n"

        await query.edit_message_text(
            history_text,
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_back_button()
        )
    elif data == "back_to_menu":
        await query.edit_message_text(
            "🏠 <b>Главное меню</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_main_menu()
        )
    elif data == "help":
        help_text = """
        <b>📚 Помощь по использованию бота</b>

        <b>Основные команды:</b>
        /start - Запустить бота
        /help - Показать эту справку
        /loan - Рассчитать кредит
        /deposit - Рассчитать вклад
        /investment - Рассчитать инвестиции
        /analysis - Анализ финансовых данных
        /settings - Настройки
        """

        await query.edit_message_text(
            help_text,
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_back_button()
        )

    return ConversationHandler.END


async def handle_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        amount = float(update.message.text.replace(',', '.'))
        if amount <= 0:
            raise ValueError

        context.user_data['amount'] = amount

        if context.user_data.get('calc_type') == 'investment':
            await update.message.reply_text(
                "Введите сумму ежемесячного пополнения (например: 5000):"
            )
            return INVESTMENT
        else:
            await update.message.reply_text(
                "Введите срок в годах (например: 5):"
            )
            return YEARS
    except:
        await update.message.reply_text(
            "❌ Неверный формат суммы. Пожалуйста, введите положительное число."
        )
        return AMOUNT


async def handle_years(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        years = float(update.message.text.replace(',', '.'))
        if years <= 0 or years > 50:
            await update.message.reply_text("Срок должен быть от 1 до 50 лет")
            return YEARS

        context.user_data['years'] = years
        await update.message.reply_text(
            "Введите годовую процентную ставку (например: 15 для 15%):"
        )
        return RATE
    except:
        await update.message.reply_text(
            "❌ Неверный формат срока. Пожалуйста, введите число."
        )
        return YEARS


async def handle_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rate = float(update.message.text.replace(',', '.'))
        if rate <= 0 or rate > 100:
            await update.message.reply_text("Ставка должна быть от 0.1 до 100%")
            return RATE

        data = context.user_data
        calc_type = data.get('calc_type')
        amount = data.get('amount')
        years = data.get('years')

        calculator = FinancialCalculator()

        if calc_type == 'loan':
            loan_type = data.get('loan_type', 'annuity')
            result = calculator.calculate_loan(amount, years, rate, loan_type)

            if not result['success']:
                await update.message.reply_text(f"❌ Ошибка расчета: {result['error']}")
                await show_main_menu(update, context)
                return ConversationHandler.END

            user_id = update.effective_user.id
            if user_id in calculation_history:
                record = CalculationRecord(
                    calc_type=CalculationType.LOAN,
                    params={"amount": amount, "years": years, "rate": rate, "type": loan_type},
                    result=result
                )
                calculation_history[user_id].append(record)

            if loan_type == "annuity":
                response = f"""
                ✅ <b>Результаты расчета аннуитетного кредита:</b>

                📊 <b>Основные параметры:</b>
                • Сумма кредита: {amount:,.2f} руб.
                • Срок: {years} лет ({result['months']} месяцев)
                • Ставка: {rate}% годовых

                💰 <b>Результаты:</b>
                • Ежемесячный платеж: {result['monthly_payment']:,.2f} руб.
                • Общая сумма выплат: {result['total_payment']:,.2f} руб.
                • Переплата: {result['overpayment']:,.2f} руб.
                • Переплата в %: {result['overpayment_percent']}%

                📅 <b>Первые 6 месяцев:</b>
                """

                for month in result['schedule']:
                    response += f"\nМесяц {month['month']}: {month['payment']:,.2f} руб. "
                    response += f"(осн.долг: {month['principal']:,.2f}, "
                    response += f"проценты: {month['interest']:,.2f})"
            else:
                response = f"""
                ✅ <b>Результаты расчета дифференцированного кредита:</b>

                📊 <b>Основные параметры:</b>
                • Сумма кредита: {amount:,.2f} руб.
                • Срок: {years} лет ({result['months']} месяцев)
                • Ставка: {rate}% годовых

                💰 <b>Результаты:</b>
                • Первый платеж: {result['first_payment']:,.2f} руб.
                • Последний платеж: ~{result['last_payment']:,.2f} руб.
                • Общая сумма выплат: ~{result['total_payment']:,.2f} руб.
                • Переплата: ~{result['overpayment']:,.2f} руб.
                • Переплата в %: {result['overpayment_percent']}%

                📅 <b>Первые 6 месяцев:</b>
                """

                for month in result['schedule']:
                    response += f"\nМесяц {month['month']}: {month['payment']:,.2f} руб. "
                    response += f"(осн.долг: {month['principal']:,.2f}, "
                    response += f"проценты: {month['interest']:,.2f})"

        elif calc_type == 'deposit':
            capitalization = data.get('capitalization', 'monthly')
            result = calculator.calculate_deposit(amount, years, rate, capitalization)

            if not result['success']:
                await update.message.reply_text(f"❌ Ошибка расчета: {result['error']}")
                await show_main_menu(update, context)
                return ConversationHandler.END

            user_id = update.effective_user.id
            if user_id in calculation_history:
                record = CalculationRecord(
                    calc_type=CalculationType.DEPOSIT,
                    params={"amount": amount, "years": years, "rate": rate, "capitalization": capitalization},
                    result=result
                )
                calculation_history[user_id].append(record)

            cap_names = {
                'monthly': 'ежемесячной',
                'quarterly': 'ежеквартальной',
                'yearly': 'ежегодной',
                'end': 'без капитализации'
            }

            response = f"""
            ✅ <b>Результаты расчета вклада с {cap_names[capitalization]}:</b>

            📊 <b>Основные параметры:</b>
            • Сумма вклада: {amount:,.2f} руб.
            • Срок: {years} лет
            • Ставка: {rate}% годовых
            • Капитализация: {capitalization}

            💰 <b>Результаты:</b>
            • Итоговая сумма: {result['final_amount']:,.2f} руб.
            • Начисленные проценты: {result['interest']:,.2f} руб.
            • Налог: {result['tax']:,.2f} руб.
            • Чистая прибыль: {result['interest'] - result['tax']:,.2f} руб.
            """

        await update.message.reply_text(
            response,
            parse_mode=ParseMode.HTML
        )
        await show_main_menu(update, context)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        await show_main_menu(update, context)

    return ConversationHandler.END


async def handle_investment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        monthly = float(update.message.text.replace(',', '.'))
        if monthly < 0:
            raise ValueError

        context.user_data['monthly'] = monthly
        await update.message.reply_text(
            "Введите срок инвестирования в годах (например: 10):"
        )
        return YEARS
    except:
        await update.message.reply_text(
            "❌ Неверный формат суммы. Пожалуйста, введите положительное число или 0."
        )
        return INVESTMENT


async def handle_investment_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        rate = float(update.message.text.replace(',', '.'))
        if rate < -100:
            await update.message.reply_text("Доходность не может быть меньше -100%")
            return RATE

        data = context.user_data
        amount = data.get('amount')
        monthly = data.get('monthly')
        years = data.get('years')

        calculator = FinancialCalculator()
        result = calculator.calculate_investment(amount, monthly, years, rate)

        if not result['success']:
            await update.message.reply_text(f"❌ Ошибка расчета: {result['error']}")
            await show_main_menu(update, context)
            return ConversationHandler.END

        user_id = update.effective_user.id
        if user_id in calculation_history:
            record = CalculationRecord(
                calc_type=CalculationType.INVESTMENT,
                params={"initial": amount, "monthly": monthly, "years": years, "rate": rate},
                result=result
            )
            calculation_history[user_id].append(record)

        response = f"""
        ✅ <b>Результаты расчета инвестиций:</b>

        📊 <b>Основные параметры:</b>
        • Начальная сумма: {amount:,.2f} руб.
        • Ежемесячное пополнение: {monthly:,.2f} руб.
        • Срок: {years} лет
        • Ожидаемая доходность: {rate}% годовых

        💰 <b>Результаты:</b>
        • Итоговая сумма: {result['final_amount']:,.2f} руб.
        • Всего вложено: {result['total_invested']:,.2f} руб.
        • Прибыль: {result['total_profit']:,.2f} руб.
        • Доходность: {result['profit_percent']}%

        📈 <b>По годам:</b>
        """

        for year_data in result['yearly_results']:
            response += f"\nГод {year_data['year']}: {year_data['amount']:,.2f} руб. "
            response += f"(вложено: {year_data['invested']:,.2f}, "
            response += f"прибыль: {year_data['profit']:,.2f})"

        await update.message.reply_text(
            response,
            parse_mode=ParseMode.HTML
        )
        await show_main_menu(update, context)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        await show_main_menu(update, context)

    return ConversationHandler.END


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text(
            "Выберите следующее действие:",
            reply_markup=Keyboards.get_main_menu()
        )
    else:
        await update.callback_query.message.reply_text(
            "Выберите следующее действие:",
            reply_markup=Keyboards.get_main_menu()
        )


async def perform_data_analysis(message, context):
    try:
        analyzer = DataAnalyzer()

        df = analyzer.generate_sample_data()
        data_info = f"""
        📈 <b>Сгенерированные финансовые данные:</b>

        • Период: {df['date'].min().date()} - {df['date'].max().date()}
        • Количество записей: {len(df)}
        • Показатели: выручка, расходы, прибыль, инвестиции

        <b>Основная статистика:</b>
        • Средняя прибыль: {df['profit'].mean():,.2f}
        • Максимальная прибыль: {df['profit'].max():,.2f}
        • Минимальная прибыль: {df['profit'].min():,.2f}
        • Стандартное отклонение: {df['profit'].std():,.2f}
        """

        await message.reply_text(
            data_info,
            parse_mode=ParseMode.HTML
        )

        await message.reply_text("🖼️ <b>Создаю визуализации...</b>", parse_mode=ParseMode.HTML)
        images = analyzer.create_visualizations(df)

        captions = [
            "📈 <b>График 1:</b> Динамика прибыли по месяцам",
            "📊 <b>График 2:</b> Распределение процентных ставок",
            "🔗 <b>График 3:</b> Корреляционная матрица",
            "📦 <b>График 4:</b> Распределение прибыли по годам"
        ]

        for i, img_buf in enumerate(images):
            await message.reply_photo(
                photo=InputFile(img_buf, filename=f"chart_{i + 1}.png"),
                caption=captions[i] if i < len(captions) else f"График {i + 1}",
                parse_mode=ParseMode.HTML
            )

        await message.reply_text("🧪 <b>Проверяю статистическую гипотезу...</b>", parse_mode=ParseMode.HTML)
        hypothesis_result = analyzer.test_statistical_hypothesis(df)

        if hypothesis_result['success']:
            hypothesis_text = f"""
            📊 <b>Результаты проверки гипотезы:</b>

            <b>Гипотеза:</b> Прибыль имеет нормальное распределение
            <b>Тест:</b> {hypothesis_result['test_name']}

            <b>Результаты:</b>
            • Статистика теста: {hypothesis_result['statistic']}
            • p-value: {hypothesis_result['p_value']}
            • Уровень значимости: 0.05
            • Вывод: распределение {hypothesis_result['interpretation']}

            <b>Дополнительные статистики:</b>
            • Асимметрия (skewness): {hypothesis_result['skewness']}
            • Эксцесс (kurtosis): {hypothesis_result['kurtosis']}
            • Среднее: {hypothesis_result['mean']:,.2f}
            • Стандартное отклонение: {hypothesis_result['std']:,.2f}
            • Размер выборки: {hypothesis_result['sample_size']}
            """

            await message.reply_text(
                hypothesis_text,
                parse_mode=ParseMode.HTML
            )

            if hypothesis_result.get('qq_plot'):
                await message.reply_photo(
                    photo=InputFile(hypothesis_result['qq_plot'], filename="qq_plot.png"),
                    caption="📈 <b>Q-Q Plot:</b> График для проверки нормальности распределения",
                    parse_mode=ParseMode.HTML
                )
        else:
            await message.reply_text(
                f"❌ Ошибка при проверке гипотезы: {hypothesis_result['error']}",
                parse_mode=ParseMode.HTML
            )

        await message.reply_text(
            "✅ <b>Анализ завершен!</b>\n\nВыберите следующее действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=Keyboards.get_main_menu()
        )

    except Exception as e:
        await message.reply_text(
            f"❌ Ошибка при анализе данных: {str(e)}",
            parse_mode=ParseMode.HTML
        )
        await message.reply_text(
            "Выберите действие:",
            reply_markup=Keyboards.get_main_menu()
        )


async def handle_quick_calc(update: Update, context: ContextTypes.DEFAULT_TYPE):

    try:
        text = update.message.text
        parts = text.split()

        if len(parts) == 3:
            amount, years, rate = map(float, [p.replace(',', '.') for p in parts])

            if amount > 100000 and rate < 30:
                calculator = FinancialCalculator()
                result = calculator.calculate_loan(amount, years, rate)

                if result['success']:
                    response = f"""
                    ✅ <b>Быстрый расчет кредита:</b>

                    Параметры:
                    • Сумма: {amount:,.0f} руб.
                    • Срок: {years} лет
                    • Ставка: {rate}%

                    Результат:
                    • Ежемесячный платеж: {result['monthly_payment']:,.2f} руб.
                    • Общая выплата: {result['total_payment']:,.2f} руб.
                    • Переплата: {result['overpayment']:,.2f} руб.
                    """
                    await update.message.reply_text(
                        response,
                        parse_mode=ParseMode.HTML,
                        reply_markup=Keyboards.get_main_menu()
                    )
            else:
                calculator = FinancialCalculator()
                result = calculator.calculate_deposit(amount, years, rate)

                if result['success']:
                    response = f"""
                    ✅ <b>Быстрый расчет вклада:</b>

                    Параметры:
                    • Сумма: {amount:,.0f} руб.
                    • Срок: {years} лет
                    • Ставка: {rate}%

                    Результат:
                    • Итоговая сумма: {result['final_amount']:,.2f} руб.
                    • Начисленные проценты: {result['interest']:,.2f} руб.
                    • Налог: {result['tax']:,.2f} руб.
                    """
                    await update.message.reply_text(
                        response,
                        parse_mode=ParseMode.HTML,
                        reply_markup=Keyboards.get_main_menu()
                    )

        elif len(parts) == 4:
            initial, monthly, years, rate = map(float, [p.replace(',', '.') for p in parts])

            calculator = FinancialCalculator()
            result = calculator.calculate_investment(initial, monthly, years, rate)

            if result['success']:
                response = f"""
                ✅ <b>Быстрый расчет инвестиций:</b>

                Параметры:
                • Начальная сумма: {initial:,.0f} руб.
                • Ежемесячное пополнение: {monthly:,.0f} руб.
                • Срок: {years} лет
                • Ожидаемая доходность: {rate}%

                Результат:
                • Итоговая сумма: {result['final_amount']:,.2f} руб.
                • Всего вложено: {result['total_invested']:,.2f} руб.
                • Прибыль: {result['total_profit']:,.2f} руб.
                • Доходность: {result['profit_percent']}%
                """
                await update.message.reply_text(
                    response,
                    parse_mode=ParseMode.HTML,
                    reply_markup=Keyboards.get_main_menu()
                )

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка быстрого расчета: {str(e)}")


async def handle_all_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text:
        await update.message.reply_text(
            "Я не понимаю эту команду. Используйте меню или команды:\n"
            "/start - Запустить бота\n"
            "/help - Помощь",
            reply_markup=Keyboards.get_main_menu()
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Произошла ошибка: {context.error}", exc_info=True)

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла ошибка. Пожалуйста, попробуйте еще раз."
        )


def main():
    print("=" * 50)
    print("Финансовый калькулятор бот")
    print("Версия: python-telegram-bot")
    print("=" * 50)

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ ОШИБКА: Токен бота не установлен!")
        print("Пожалуйста, создайте файл .env и добавьте:")
        print("BOT_TOKEN=ваш_токен_от_BotFather")
        return

    print("✅ Бот запускается...")
    print("📱 Перейдите в Telegram и найдите своего бота")
    print("⚡ Используйте /start для начала работы")
    print("=" * 50)

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))

    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler)],
        states={
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_amount)],
            YEARS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_years)],
            RATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_rate)],
            INVESTMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_investment)],
        },
        fallbacks=[CommandHandler("start", start_command)],
        allow_reentry=True
    )

    investment_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(button_handler, pattern="^investment_calc$")],
        states={
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_amount)],
            INVESTMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_investment)],
            YEARS: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_years)],
            RATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_investment_rate)],
        },
        fallbacks=[CommandHandler("start", start_command)],
        allow_reentry=True
    )

    application.add_handler(conv_handler)
    application.add_handler(investment_handler)
    application.add_handler(CallbackQueryHandler(button_handler))

    application.add_handler(MessageHandler(
        filters.Regex(r'^\d+(?:[.,]\d+)?\s+\d+(?:[.,]\d+)?\s+\d+(?:[.,]\d+)?(?:\s+\d+(?:[.,]\d+)?)?$'),
        handle_quick_calc
    ))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_messages))

    application.add_error_handler(error_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    main()