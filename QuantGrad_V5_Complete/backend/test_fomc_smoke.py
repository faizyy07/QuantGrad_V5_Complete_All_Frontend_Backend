"""Minimal local smoke test for the FOMC calendar helper."""

from macro_fetcher import fetch_fomc_dates


def main() -> None:
    result = fetch_fomc_dates()
    assert {"last_fomc", "next_fomc", "hours_until_next", "event_flag_24h"} <= result.keys()
    assert result["event_flag_24h"] in (0, 1)
    print(result)


if __name__ == "__main__":
    main()
