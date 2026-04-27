const SHANGHAI_TIMEZONE = 'Asia/Shanghai';
const SHANGHAI_OFFSET = '+08:00';

function formatDateParts(date, timezone = SHANGHAI_TIMEZONE) {
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });

  const parts = formatter.formatToParts(date);
  const lookup = Object.fromEntries(
    parts
      .filter((part) => part.type !== 'literal')
      .map((part) => [part.type, part.value]),
  );

  return {
    year: lookup.year,
    month: lookup.month,
    day: lookup.day,
    date: `${lookup.year}-${lookup.month}-${lookup.day}`,
    timezone,
  };
}

function shiftIsoDate(dateText, deltaDays) {
  const [year, month, day] = dateText.split('-').map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day + deltaDays));
  return formatDateParts(shifted, 'UTC').date;
}

function getTargetDateParts(now = new Date(), timezone = SHANGHAI_TIMEZONE) {
  return formatDateParts(now, timezone);
}

function buildResearchWindow(
  targetDate,
  { timezone = SHANGHAI_TIMEZONE, lookbackDays = 3 } = {},
) {
  if (!targetDate) {
    throw new Error('targetDate is required');
  }

  if (!Number.isInteger(lookbackDays) || lookbackDays < 1) {
    throw new Error('lookbackDays must be a positive integer');
  }

  const startDate = shiftIsoDate(targetDate, -(lookbackDays - 1));

  return {
    timezone,
    targetDate,
    lookbackDays,
    startDate,
    endDate: targetDate,
    start: `${startDate}T00:00:00${SHANGHAI_OFFSET}`,
    end: `${targetDate}T23:59:59${SHANGHAI_OFFSET}`,
  };
}

module.exports = {
  SHANGHAI_TIMEZONE,
  getTargetDateParts,
  buildResearchWindow,
};
