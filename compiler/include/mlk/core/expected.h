// MLK+ monadic helpers over Result<T> (Rule 26).
//
// std::expected already provides and_then / transform / or_else in C++23+.
// These helpers add small conveniences used pervasively by passes so that
// error propagation stays branch-minimal and readable without macros.
#pragma once

#include "mlk/core/result.h"

#include <type_traits>
#include <utility>

namespace mlk {

/// Maps a Result<T> to Result<U> via a function U(T&&), propagating errors.
template <typename T, typename Fn>
[[nodiscard]] auto mapResult(Result<T>&& r, Fn&& fn)
    -> Result<std::invoke_result_t<Fn, T&&>> {
    if (!r.has_value()) {
        return std::unexpected<Error>(std::move(r).error());
    }
    return std::invoke_result_t<Fn, T&&>(std::forward<Fn>(fn)(*std::move(r)));
}

/// Chains a function returning Result<U> onto a Result<T>, propagating errors.
template <typename T, typename Fn>
[[nodiscard]] auto andThenResult(Result<T>&& r, Fn&& fn)
    -> std::invoke_result_t<Fn, T&&> {
    using Out = std::invoke_result_t<Fn, T&&>;
    static_assert(std::is_same_v<typename Out::error_type, Error>,
                  "andThenResult requires Result<U, mlk::Error>");
    if (!r.has_value()) {
        return Out{std::unexpected<Error>(std::move(r).error())};
    }
    return std::forward<Fn>(fn)(*std::move(r));
}

}  // namespace mlk
